import unittest

from zmd.rl.envs.vascular_env import (
    DEFAULT_REWARD_PARAMS,
    compute_progress_reward,
)


def _transition(progress, last_progress, best_progress):
    return compute_progress_reward(
        progress=progress,
        last_progress=last_progress,
        best_progress=best_progress,
        reward_params=DEFAULT_REWARD_PARAMS,
    )


class ProgressRewardTest(unittest.TestCase):
    def test_progress_reward_only_pays_for_new_frontier(self):
        first, best = _transition(0.20, 0.00, 0.00)
        retreat, best = _transition(0.10, 0.20, best)
        revisit, best = _transition(0.20, 0.10, best)

        self.assertAlmostEqual(first["progress"], 40.0)
        self.assertEqual(retreat["progress"], 0.0)
        self.assertAlmostEqual(retreat["backtrack"], -10.0)
        self.assertEqual(revisit["progress"], 0.0)
        self.assertAlmostEqual(
            sum(part["progress"] for part in (first, retreat, revisit)),
            40.0,
        )
        self.assertAlmostEqual(best, 0.20)

    def test_advance_retract_cycle_has_negative_incremental_return(self):
        best = 0.30
        retreat, best = _transition(0.20, 0.30, best)
        revisit, best = _transition(0.30, 0.20, best)

        cycle_reward = sum(retreat.values()) + sum(revisit.values())
        self.assertLess(cycle_reward, 0.0)
        self.assertAlmostEqual(best, 0.30)

    def test_frontier_reward_is_bounded_by_progress_weight(self):
        best = 0.0
        total_progress_reward = 0.0
        last = 0.0
        for progress in (0.10, 0.05, 0.40, 0.20, 0.91):
            parts, best = _transition(progress, last, best)
            total_progress_reward += parts["progress"]
            last = progress

        self.assertAlmostEqual(
            total_progress_reward,
            DEFAULT_REWARD_PARAMS["progress_weight"] * 0.91,
        )
        self.assertLessEqual(
            total_progress_reward,
            DEFAULT_REWARD_PARAMS["progress_weight"],
        )


if __name__ == "__main__":
    unittest.main()
