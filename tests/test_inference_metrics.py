import unittest

from zmd.rl.inference_metrics import summarize_rollout


class InferenceMetricsTest(unittest.TestCase):
    def test_summary_counts_rotation_blocking_and_progress_auc(self):
        rows = [
            {
                "step": 1, "action_index": 0, "progress": 0.20,
                "requested_push_mm": 0.75, "speed_phase": "proximal",
                "blocked_this_step": 1, "wall_slide_attempted": 1,
                "wall_slide_success": 0, "safety_margin_mm": 0.40,
                "distance_to_end_mm": 8.0, "reward": 1.0,
            },
            {
                "step": 2, "action_index": 2, "progress": 0.40,
                "requested_push_mm": 0.0, "speed_phase": "proximal",
                "blocked_this_step": 0, "wall_slide_attempted": 0,
                "wall_slide_success": 0, "safety_margin_mm": 0.25,
                "distance_to_end_mm": 6.0, "reward": -0.1,
            },
            {
                "step": 3, "action_index": 3, "progress": 0.80,
                "requested_push_mm": 0.0, "speed_phase": "distal",
                "blocked_this_step": 0, "wall_slide_attempted": 0,
                "wall_slide_success": 0, "safety_margin_mm": 0.30,
                "distance_to_end_mm": 2.0, "reward": -0.1,
            },
        ]
        summary = summarize_rollout(
            rows,
            success=False,
            termination_reason="max_steps",
            tick_ms=90,
        )
        self.assertEqual(summary["advance_count"], 1)
        self.assertEqual(summary["rotation_count"], 2)
        self.assertAlmostEqual(summary["rotation_rate"], 2 / 3)
        self.assertEqual(summary["blocked_advance_count"], 1)
        self.assertEqual(summary["blocked_advance_rate"], 1.0)
        self.assertAlmostEqual(summary["progress_auc_step_normalized"], 0.45)
        self.assertAlmostEqual(summary["minimum_safety_margin_mm"], 0.25)
        self.assertAlmostEqual(summary["estimated_execution_time_s"], 0.27)

if __name__ == "__main__":
    unittest.main()
