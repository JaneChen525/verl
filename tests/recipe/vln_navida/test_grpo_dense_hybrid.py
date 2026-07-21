import numpy as np
import torch

from recipe.vln_navida.vln_traj_grpo import (
    compute_grpo_dense_hybrid_advantage,
    compute_grpo_trajectory_advantage,
)


def _batch(dense_returns):
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )
    token_level_rewards = torch.zeros(5, 3)
    for row, score in enumerate(dense_returns):
        if response_mask[row].any():
            last = response_mask[row].nonzero()[-1].item()
            token_level_rewards[row, last] = score

    return {
        "token_level_rewards": token_level_rewards,
        "response_mask": response_mask,
        "index": np.array(["episode", "episode", "episode", "episode", "padding"], dtype=object),
        "trajectory_index": np.array(["success", "success", "failure", "failure", "padding"], dtype=object),
        "trajectory_rewards": np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "decision_loss_weight": np.array([0.5, 0.5, 0.5, 0.5, 0.0], dtype=np.float32),
    }


def _sparse_rewards(batch):
    rewards = torch.zeros_like(batch["token_level_rewards"])
    for row, score in enumerate(batch["trajectory_rewards"]):
        if batch["response_mask"][row].any():
            last = batch["response_mask"][row].nonzero()[-1].item()
            rewards[row, last] = float(score)
    return rewards


def test_lambda_zero_matches_trajectory_grpo():
    batch = _batch([3.0, 1.0, 2.0, 2.0, 0.0])
    hybrid, hybrid_returns = compute_grpo_dense_hybrid_advantage(
        **batch, dense_lambda=0.0
    )
    expected, _ = compute_grpo_trajectory_advantage(
        token_level_rewards=_sparse_rewards(batch),
        response_mask=batch["response_mask"],
        index=batch["index"],
        trajectory_index=batch["trajectory_index"],
    )

    torch.testing.assert_close(hybrid, expected)
    torch.testing.assert_close(hybrid_returns, expected)


def test_dense_component_is_trajectory_shift_invariant_and_padding_safe():
    batch = _batch([3.0, 1.0, 1.0, 3.0, 0.0])
    shifted = _batch([13.0, 11.0, -4.0, -2.0, 0.0])

    actual, _ = compute_grpo_dense_hybrid_advantage(
        **batch, dense_lambda=0.2
    )
    shifted_actual, _ = compute_grpo_dense_hybrid_advantage(
        **shifted, dense_lambda=0.2
    )

    torch.testing.assert_close(actual, shifted_actual)
    assert torch.count_nonzero(actual[-1]) == 0

    grpo = 1 / np.sqrt(2)
    expected_scores = torch.tensor(
        [grpo + 0.2, grpo - 0.2, -grpo - 0.2, -grpo + 0.2],
        dtype=actual.dtype,
    )
    for row, expected in enumerate(expected_scores):
        mask = batch["response_mask"][row].to(torch.bool)
        torch.testing.assert_close(actual[row, mask], expected.expand(int(mask.sum())))


def test_clipping_preserves_zero_mean_within_each_trajectory():
    response_mask = torch.ones(6, 1, dtype=torch.long)
    batch = {
        "token_level_rewards": torch.tensor(
            [[100.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        ),
        "response_mask": response_mask,
        "index": np.array(["episode"] * 6, dtype=object),
        "trajectory_index": np.array(["a", "a", "a", "b", "b", "b"], dtype=object),
        "trajectory_rewards": np.zeros(6, dtype=np.float32),
        "decision_loss_weight": np.full(6, 1 / 3, dtype=np.float32),
    }

    actual, _ = compute_grpo_dense_hybrid_advantage(
        **batch, dense_lambda=1.0, dense_clip=1.0
    )
    scores = actual[:, 0]
    torch.testing.assert_close(scores[:3].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(scores[3:].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    assert scores.abs().max() <= 1.0
    assert scores[0] > 0
    assert scores[1] < 0


def test_all_same_success_uses_only_balanced_dense_credit():
    batch = _batch([3.0, 1.0, 1.0, 3.0, 0.0])
    batch["trajectory_rewards"][:4] = 1.0
    actual, _ = compute_grpo_dense_hybrid_advantage(
        **batch, dense_lambda=0.2
    )

    scores = actual[:4, 0]
    torch.testing.assert_close(scores, torch.tensor([0.2, -0.2, -0.2, 0.2]))
    torch.testing.assert_close(scores[:2].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(scores[2:].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)


def test_dense_rms_gives_unequal_length_trajectories_equal_weight():
    dense_returns = torch.tensor([2.0, 0.0, 3.0, 1.0, 1.0, 1.0])
    batch = {
        "token_level_rewards": dense_returns.unsqueeze(-1),
        "response_mask": torch.ones(6, 1, dtype=torch.long),
        "index": np.array(["episode"] * 6, dtype=object),
        "trajectory_index": np.array(["short", "short", "long", "long", "long", "long"], dtype=object),
        "trajectory_rewards": np.zeros(6, dtype=np.float32),
        "decision_loss_weight": np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25], dtype=np.float32),
    }

    actual, _ = compute_grpo_dense_hybrid_advantage(
        **batch, dense_lambda=1.0, dense_clip=3.0
    )
    equal_trajectory_rms = np.sqrt((1.0 + 0.75) / 2)
    expected = torch.tensor([1.0, -1.0, 1.5, -0.5, -0.5, -0.5])
    expected = expected / equal_trajectory_rms
    torch.testing.assert_close(actual[:, 0], expected, atol=2e-6, rtol=1e-6)
