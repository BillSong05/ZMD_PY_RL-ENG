import unittest

from zmd.rl.speed_schedule import build_speed_config, resolve_speed_config


class SegmentedSpeedConfigTest(unittest.TestCase):
    def test_canonical_profiles(self):
        expected = {
            "B0": (0.50, 0.50, 0.80),
            "A1": (0.75, 0.50, 0.80),
            "A2": (1.00, 0.50, 0.80),
            "A3": (0.75, 0.25, 0.80),
            "A4": (1.50, 0.50, 0.80),
            "A5": (2.00, 0.50, 0.80),
            "A6": (2.25, 0.50, 0.80),
            "A7": (2.50, 0.50, 0.80),
            "P75": (2.00, 0.50, 0.75),
            "P85": (2.00, 0.50, 0.85),
            "P90": (2.00, 0.50, 0.90),
            "P95": (2.00, 0.50, 0.95),
        }
        for name, (proximal, distal, switch) in expected.items():
            cfg = resolve_speed_config(name)
            self.assertEqual(
                (cfg.proximal_push_mm, cfg.distal_push_mm),
                (proximal, distal),
            )
            self.assertEqual(cfg.switch_progress, switch)

    def test_switch_uses_best_progress_and_can_be_latched(self):
        cfg = resolve_speed_config("A3")
        self.assertEqual(cfg.phase(0.79), "proximal")
        self.assertEqual(cfg.push_step_mm(0.79), 0.75)
        self.assertEqual(cfg.phase(0.80), "distal")
        self.assertEqual(cfg.push_step_mm(0.80), 0.25)
        self.assertEqual(cfg.phase(0.20, distal_latched=True), "distal")

    def test_override_is_labeled_custom(self):
        cfg = build_speed_config("A1", proximal_push_mm=0.90)
        self.assertEqual(cfg.profile, "CUSTOM")
        self.assertEqual(cfg.proximal_push_mm, 0.90)
        self.assertEqual(cfg.distal_push_mm, 0.50)

    def test_invalid_custom_values_are_rejected(self):
        with self.assertRaises(ValueError):
            build_speed_config("CUSTOM", proximal_push_mm=0.0)
        with self.assertRaises(ValueError):
            build_speed_config("CUSTOM", switch_progress=1.1)


if __name__ == "__main__":
    unittest.main()
