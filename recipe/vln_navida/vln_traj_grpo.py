"""Trajectory-level GRPO advantage.

Standard GRPO groups all decisions by uid (episode). With flatten, a group has
N_decisions × M_rollouts samples — longer trajectories get more votes in the
group mean/std, biasing advantage toward them.

This estimator deduplicates by trajectory_uid first: each trajectory contributes
exactly one reward to the group statistics, then the per-trajectory advantage is
broadcast to all its decisions.

    μ_g = (1/|T_g|) Σ_{t∈T_g} r_t
    A_t = (r_t − μ_g) / (σ_g + ε)

All decisions of trajectory t receive the same A_t.
"""

import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("p15_dense_return")
def compute_p15_dense_return_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the precomputed decision return directly as A=G, without normalization."""
    del kwargs
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        returns = scores.unsqueeze(-1) * response_mask
        advantages = returns.clone()
    return advantages, returns


@register_adv_est("grpo_trajectory")
def compute_grpo_trajectory_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        bsz = scores.shape[0]

        # Step 1: deduplicate by trajectory_uid, verify reward consistency
        group_traj: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        for i in range(bsz):
            g, t = index[i], trajectory_index[i]
            if t in group_traj[g]:
                if not torch.isclose(group_traj[g][t], scores[i], atol=1e-5):
                    raise ValueError(
                        f"Inconsistent rewards for trajectory {t}: "
                        f"{group_traj[g][t].item():.4f} vs {scores[i].item():.4f}"
                    )
            else:
                group_traj[g][t] = scores[i]

        # Step 2: per-group mean/std from M unique trajectory rewards
        id2mean: dict[str, torch.Tensor] = {}
        id2std: dict[str, torch.Tensor] = {}
        for g, traj_dict in group_traj.items():
            rs = torch.stack(list(traj_dict.values()))
            if len(rs) == 1:
                id2mean[g] = scores.new_zeros(())
                id2std[g] = scores.new_ones(())
            else:
                id2mean[g] = rs.mean()
                id2std[g] = rs.std()

        # Step 3: normalize and broadcast to token dimension
        for i in range(bsz):
            g = index[i]
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[g]) / (id2std[g] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[g]
        advantages = scores.unsqueeze(-1) * response_mask

    return advantages, advantages


@register_adv_est("grpo_dense_hybrid")
def compute_grpo_dense_hybrid_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_index: np.ndarray,
    trajectory_rewards: np.ndarray,
    decision_loss_weight: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    dense_lambda: Optional[float] = None,
    dense_clip: Optional[float] = None,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine trajectory GRPO with centered, normalized P15 decision returns.

    The sparse trajectory reward determines which rollout is better inside an
    episode group.  The dense component has zero mean within each trajectory,
    so it only redistributes credit among that trajectory's decisions.
    """
    del config, kwargs
    if dense_lambda is None:
        dense_lambda = float(os.environ.get("VLN_DENSE_LAMBDA", "0.2"))
    if dense_clip is None:
        dense_clip = float(os.environ.get("VLN_DENSE_CLIP", "3.0"))
    if not np.isfinite(dense_lambda) or dense_lambda < 0:
        raise ValueError(f"VLN_DENSE_LAMBDA must be non-negative, got {dense_lambda}")
    if not np.isfinite(dense_clip) or dense_clip <= 0:
        raise ValueError(f"VLN_DENSE_CLIP must be positive, got {dense_clip}")

    with torch.no_grad():
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        bsz, response_length = token_level_rewards.shape
        valid = response_mask.to(torch.bool).any(dim=-1)

        if len(index) != bsz or len(trajectory_index) != bsz:
            raise ValueError("uid and trajectory_uid must match the decision batch size")

        trajectory_rewards_t = torch.as_tensor(
            trajectory_rewards, device=device, dtype=dtype
        )
        decision_weights = torch.as_tensor(
            decision_loss_weight, device=device, dtype=dtype
        )
        if trajectory_rewards_t.numel() != bsz or decision_weights.numel() != bsz:
            raise ValueError(
                "trajectory_reward and decision_loss_weight must match the decision batch size"
            )
        if not torch.isfinite(trajectory_rewards_t[valid]).all():
            raise ValueError("trajectory_reward must be finite for valid decisions")
        if not torch.isfinite(decision_weights[valid]).all() or (
            decision_weights[valid] < 0
        ).any():
            raise ValueError("decision_loss_weight must be finite and non-negative")
        if valid.any() and decision_weights[valid].sum() <= 0:
            raise ValueError("valid decisions must have positive total decision_loss_weight")

        # Reuse the P13 trajectory-GRPO implementation by placing each sparse
        # trajectory reward at the last valid response token of every decision.
        sparse_token_rewards = torch.zeros_like(token_level_rewards)
        token_positions = torch.arange(response_length, device=device).unsqueeze(0)
        last_valid = torch.where(
            response_mask.to(torch.bool), token_positions, token_positions.new_full((), -1)
        ).max(dim=-1).values
        rows = torch.arange(bsz, device=device)[valid]
        sparse_token_rewards[rows, last_valid[valid]] = trajectory_rewards_t[valid]
        grpo_tokens, _ = compute_grpo_trajectory_advantage(
            token_level_rewards=sparse_token_rewards,
            response_mask=response_mask,
            index=index,
            trajectory_index=trajectory_index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )

        # Every response token of one decision has the same GRPO advantage.
        first_valid = response_mask.to(torch.bool).to(torch.int64).argmax(dim=-1)
        grpo_scores = grpo_tokens.gather(1, first_valid.unsqueeze(-1)).squeeze(-1)
        grpo_scores = grpo_scores * valid

        dense_returns = token_level_rewards.sum(dim=-1)
        if not torch.isfinite(dense_returns[valid]).all():
            raise ValueError("dense decision returns must be finite")
        dense_centered = torch.zeros_like(dense_returns)
        valid_np = valid.detach().cpu().numpy()
        for trajectory_uid in dict.fromkeys(trajectory_index[valid_np].tolist()):
            trajectory_mask_np = (trajectory_index == trajectory_uid) & valid_np
            trajectory_mask = torch.as_tensor(trajectory_mask_np, device=device)
            trajectory_values = dense_returns[trajectory_mask]
            dense_centered[trajectory_mask] = trajectory_values - trajectory_values.mean()

        # decision_loss_weight is 1 / num_decisions, so this RMS gives each
        # trajectory equal weight even when trajectory lengths differ.
        valid_weights = decision_weights * valid
        dense_scale = torch.sqrt(
            (valid_weights * dense_centered.square()).sum()
            / valid_weights.sum().clamp_min(epsilon)
        )
        dense_normalized = dense_centered / (dense_scale + epsilon)
        dense_clip_fraction = (
            (dense_normalized[valid].abs() > dense_clip).float().mean()
            if valid.any()
            else dense_normalized.new_zeros(())
        )
        dense_scores = torch.clamp(dense_normalized, min=-dense_clip, max=dense_clip)
        # Clipping a skewed trajectory can reintroduce a non-zero mean. Project
        # it back to the zero-mean subspace so dense credit cannot add a net
        # trajectory-level preference.
        for trajectory_uid in dict.fromkeys(trajectory_index[valid_np].tolist()):
            trajectory_mask_np = (trajectory_index == trajectory_uid) & valid_np
            trajectory_mask = torch.as_tensor(trajectory_mask_np, device=device)
            dense_scores[trajectory_mask] -= dense_scores[trajectory_mask].mean()
            trajectory_max = dense_scores[trajectory_mask].abs().max()
            if trajectory_max > dense_clip:
                dense_scores[trajectory_mask] *= dense_clip / trajectory_max
        dense_scores = dense_scores * valid

        hybrid_scores = grpo_scores + dense_lambda * dense_scores
        advantages = hybrid_scores.unsqueeze(-1) * response_mask

        if valid.any():
            grpo_rms = grpo_scores[valid].square().mean().sqrt().item()
            dense_rms = dense_scores[valid].square().mean().sqrt().item()
            hybrid_mean = hybrid_scores[valid].mean().item()
            hybrid_rms = hybrid_scores[valid].square().mean().sqrt().item()
            print(
                "[VLN hybrid advantage] "
                f"lambda={dense_lambda:.3f}, grpo_rms={grpo_rms:.3f}, "
                f"dense_rms={dense_rms:.3f}, clip_frac={dense_clip_fraction.item():.3%}, "
                f"hybrid_mean={hybrid_mean:.3f}, "
                f"hybrid_rms={hybrid_rms:.3f}"
            )

    # There is no critic target for this hybrid estimator; match trajectory
    # GRPO semantics and expose the combined advantage as returns as well.
    return advantages, advantages
