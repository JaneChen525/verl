import asyncio
import unittest

from recipe.vln_navida.full_episode_agent_loop import DecisionGen, run_episode


class FakeEnv:
    def __init__(self):
        self.instruction = "find the goal"
        self._metrics = {"distance_to_goal": 5.0, "success": 0.0}
        self._step = 0

    async def reset(self, extra_info):
        return "frame-0"

    def metrics(self):
        return dict(self._metrics)

    async def step(self, actions):
        assert len(actions) == 1
        self._step += 1
        if self._step == 1:
            self._metrics = {"distance_to_goal": 4.75, "success": 0.0}
            done = False
        elif self._step == 2:
            self._metrics = {"distance_to_goal": 4.5, "success": 0.0}
            done = False
        else:
            self._metrics = {"distance_to_goal": 4.5, "success": 1.0}
            done = True
        return {"done": done, "metrics": dict(self._metrics)}

    def current_jpeg_b64(self):
        return f"frame-{self._step}"


class P15DenseRewardTest(unittest.TestCase):
    def test_atomic_step_discounting(self):
        actions = iter(["<answer>forward 50</answer>", "<answer>stop</answer>"])

        async def decide(instruction, history_window, all_frames):
            return DecisionGen(action_text=next(actions))

        trajectory = asyncio.run(
            run_episode(
                FakeEnv(),
                {"episode_id": "1", "scene_id": "scene"},
                decide,
                group_uid="group",
                trajectory_uid="trajectory",
                reward_mode="p15_dense",
                dense_gamma=0.95,
            )
        )

        first, stop = trajectory.decisions
        self.assertAlmostEqual(first.decision_reward, 0.24 + 0.95 * 0.24)
        self.assertAlmostEqual(first.discount_to_next, 0.95**2)
        self.assertAlmostEqual(stop.decision_reward, 2.49)
        self.assertAlmostEqual(stop.decision_return, 2.49)
        self.assertAlmostEqual(
            first.decision_return,
            (0.24 + 0.95 * 0.24) + 0.95**2 * 2.49,
        )
        self.assertEqual(trajectory.reward, 1.0)

    def test_unexecuted_stop_is_not_marked_as_stop(self):
        class MaxStepEnv(FakeEnv):
            async def step(self, actions):
                self._step += 1
                self._metrics = {"distance_to_goal": 4.75, "success": 0.0}
                return {
                    "done": True,
                    "metrics": dict(self._metrics),
                    "executed_actions": [actions[0]],
                }

        async def decide(instruction, history_window, all_frames):
            return DecisionGen(action_text="<answer>forward 25</answer>, <answer>stop</answer>")

        trajectory = asyncio.run(
            run_episode(
                MaxStepEnv(),
                {"episode_id": "1", "scene_id": "scene"},
                decide,
                group_uid="group",
                trajectory_uid="trajectory",
            )
        )

        self.assertEqual(trajectory.decisions[0].atomic_chunk, [1])
        self.assertFalse(trajectory.decisions[0].is_stop_action)


if __name__ == "__main__":
    unittest.main()
