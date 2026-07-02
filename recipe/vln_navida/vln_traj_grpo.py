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

from collections import defaultdict

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


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
