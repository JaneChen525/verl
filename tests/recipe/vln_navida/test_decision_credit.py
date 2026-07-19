import unittest

from recipe.vln_navida.decision_credit import assign_success_buffer_credit


def _row(
    uid,
    trajectory_uid,
    turn_id,
    start,
    end,
    *,
    success=False,
    reward=0.1,
    decision_return=0.5,
    start_heading=(1.0, 0.0),
    end_heading=(1.0, 0.0),
    stop=False,
    start_distance=5.0,
    end_distance=4.0,
):
    return {
        "uid": uid,
        "trajectory_uid": trajectory_uid,
        "turn_id": turn_id,
        "trajectory_success": float(success),
        "decision_reward": reward,
        "decision_return": decision_return,
        "training_score": -99.0,
        "start_position": list(start),
        "end_position": list(end),
        "start_heading": list(start_heading),
        "end_heading": list(end_heading),
        "is_stop_action": stop,
        "start_distance": start_distance,
        "end_distance": end_distance,
    }


def _successful_route(uid="episode", trajectory_uid="success"):
    return [
        _row(uid, trajectory_uid, turn, (turn, 0, 0), (turn + 1, 0, 0), success=True)
        for turn in range(4)
    ]


class DecisionCreditTest(unittest.TestCase):
    def test_wrong_turn_gets_credit_and_post_tail_is_zero(self):
        success = _successful_route()
        failure = [
            _row("episode", "failure", 0, (0, 0, 0), (1, 0, 0), reward=0.2),
            _row(
                "episode",
                "failure",
                1,
                (1, 0, 0),
                (1, 0, 2),
                reward=-0.2,
                end_heading=(0, 1),
            ),
            _row(
                "episode",
                "failure",
                2,
                (1, 0, 2),
                (1, 0, 3),
                reward=-0.1,
                start_heading=(0, 1),
                end_heading=(0, 1),
            ),
            _row(
                "episode",
                "failure",
                3,
                (1, 0, 3),
                (1, 0, 4),
                reward=-0.1,
                start_heading=(0, 1),
                end_heading=(0, 1),
            ),
        ]
        rows = [failure[2], success[1], failure[0], success[3], failure[3], success[0], failure[1], success[2]]

        metrics = assign_success_buffer_credit(rows)

        ordered_failure = sorted(failure, key=lambda row: row["turn_id"])
        self.assertGreater(ordered_failure[1]["credit_weight"], 0.95)
        self.assertLess(ordered_failure[1]["training_score"], -1.0)
        self.assertEqual(ordered_failure[1]["credit_region"], "key")
        self.assertEqual(ordered_failure[2]["training_score"], 0.0)
        self.assertEqual(ordered_failure[3]["training_score"], 0.0)
        self.assertAlmostEqual(sum(row["credit_weight"] for row in failure), 1.0)
        self.assertEqual(metrics["vln/credit/persistent_exit_trajectories"], 1.0)

    def test_failed_stop_inside_buffer_is_the_key(self):
        rows = _successful_route() + [
            _row("episode", "stop", 0, (0, 0, 0), (1, 0, 0)),
            _row(
                "episode",
                "stop",
                1,
                (1, 0, 0),
                (1, 0, 0),
                reward=-0.01,
                stop=True,
            ),
        ]

        metrics = assign_success_buffer_credit(rows)
        stopped = [row for row in rows if row["trajectory_uid"] == "stop"]

        self.assertEqual(stopped[0]["credit_weight"], 0.0)
        self.assertEqual(stopped[1]["credit_weight"], 1.0)
        self.assertAlmostEqual(stopped[1]["training_score"], -1.01)
        self.assertEqual(metrics["vln/credit/early_stop_trajectories"], 1.0)

    def test_heading_mismatch_is_charged_at_the_turn_onset(self):
        success = _successful_route()
        failure = [
            _row(
                "episode",
                "pure-turn",
                0,
                (0, 0, 0),
                (0, 0, 0),
                start_heading=(1, 0),
                end_heading=(0, 1),
            ),
            _row(
                "episode",
                "pure-turn",
                1,
                (0, 0, 0),
                (0, 0, 0),
                start_heading=(0, 1),
                end_heading=(0, 1),
            ),
            _row(
                "episode",
                "pure-turn",
                2,
                (0, 0, 0),
                (0, 0, 2),
                start_heading=(0, 1),
                end_heading=(0, 1),
            ),
            _row(
                "episode",
                "pure-turn",
                3,
                (0, 0, 2),
                (0, 0, 3),
                start_heading=(0, 1),
                end_heading=(0, 1),
            ),
        ]

        assign_success_buffer_credit(success + failure)

        self.assertGreater(failure[0]["credit_criticality"], 0.0)
        self.assertEqual(failure[1]["credit_criticality"], 0.0)
        self.assertGreater(failure[0]["credit_weight"], 0.0)

    def test_in_buffer_max_step_uses_stagnation_onset(self):
        success = _successful_route()
        failure = [
            _row(
                "episode", "stagnant", 0, (0, 0, 0), (1, 0, 0),
                start_distance=5.0, end_distance=4.0,
            ),
            _row(
                "episode", "stagnant", 1, (1, 0, 0), (2, 0, 0),
                start_distance=4.0, end_distance=3.0,
            ),
            _row(
                "episode", "stagnant", 2, (2, 0, 0), (2, 0, 0),
                start_distance=3.0, end_distance=3.0,
            ),
            _row(
                "episode", "stagnant", 3, (2, 0, 0), (2, 0, 0),
                start_distance=3.0, end_distance=3.0,
            ),
        ]

        metrics = assign_success_buffer_credit(success + failure)

        self.assertEqual(failure[2]["credit_weight"], 1.0)
        self.assertEqual(failure[3]["training_score"], 0.0)
        self.assertEqual(metrics["vln/credit/stagnation_trajectories"], 1.0)

    def test_no_success_rollout_falls_back_exactly_to_p15(self):
        rows = [
            _row("episode", "failure-a", 0, (0, 0, 0), (1, 0, 0), decision_return=1.25),
            _row("episode", "failure-b", 0, (0, 0, 0), (0, 0, 1), decision_return=-0.75),
        ]

        metrics = assign_success_buffer_credit(rows)

        self.assertEqual([row["training_score"] for row in rows], [1.25, -0.75])
        self.assertTrue(all(row["credit_mode"] == "p15_fallback" for row in rows))
        self.assertEqual(metrics["vln/credit/fallback_groups"], 1.0)

    def test_uid_groups_do_not_merge_even_if_rows_are_interleaved(self):
        success_group = _successful_route(uid="uid-success")
        fallback_row = _row(
            "uid-failure",
            "failure",
            0,
            (0, 0, 0),
            (0, 0, 2),
            decision_return=0.42,
        )
        rows = [success_group[2], fallback_row, success_group[0], success_group[3], success_group[1]]

        metrics = assign_success_buffer_credit(rows)

        self.assertEqual(fallback_row["training_score"], 0.42)
        self.assertEqual(fallback_row["credit_mode"], "p15_fallback")
        self.assertEqual(metrics["vln/credit/groups"], 2.0)
        self.assertEqual(metrics["vln/credit/fallback_groups"], 1.0)


if __name__ == "__main__":
    unittest.main()
