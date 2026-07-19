import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from recipe.vln_navida.rollout_trace import export_decision_trace, sparse_01_advantages


def _row(uid, trajectory_uid, success, turn_id=0):
    return {
        "uid": uid,
        "trajectory_uid": trajectory_uid,
        "trajectory_success": float(success),
        "trajectory_reward": float(success),
        "trajectory_metrics": {"success": float(success), "num_decisions": 1},
        "turn_id": turn_id,
        "training_score": 0.25,
        "decision_return": 0.5,
        "action_text": "<answer>stop</answer>",
        "atomic_actions": [0],
        "is_stop_action": True,
    }


class RolloutTraceTest(unittest.TestCase):
    def test_sparse_01_matches_six_of_eight_grpo(self):
        rows = [_row("episode", f"trajectory-{index}", index < 6) for index in range(8)]
        advantages = sparse_01_advantages(rows)
        self.assertTrue(
            np.isclose(advantages[("episode", "trajectory-0")], 0.540060, atol=1e-5)
        )
        self.assertTrue(
            np.isclose(advantages[("episode", "trajectory-7")], -1.620182, atol=1e-5)
        )

    def test_sparse_01_all_equal_is_zero(self):
        rows = [_row("episode", f"trajectory-{index}", True) for index in range(8)]
        advantages = sparse_01_advantages(rows)
        self.assertTrue(all(np.isclose(value, 0.0) for value in advantages.values()))

    def test_export_preserves_three_advantages(self):
        rows = [_row("episode", "success", True), _row("episode", "failure", False)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            export_decision_trace(rows, {"global_steps": 3}, str(path))
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["global_step"], 3)
        self.assertEqual(records[0]["p15_return"], 0.5)
        self.assertEqual(records[0]["p16_advantage"], 0.25)
        self.assertEqual(records[0]["termination_reason"], "stop_action")


if __name__ == "__main__":
    unittest.main()
