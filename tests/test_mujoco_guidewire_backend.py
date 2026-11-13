import math
import unittest
from pathlib import Path

import numpy as np

from zmd.centerline.centerline_model import Centerline
from zmd.simulation.backends.mujoco_guidewire_backend import MujocoGuidewireBackend
from zmd.simulation.guidewire.mujoco_config import MujocoGuidewireConfig
from zmd.simulation.guidewire.mujoco_guidewire import MujocoGuidewire


class MujocoGuidewireBackendTest(unittest.TestCase):
    def make_backend(self, **cfg_overrides) -> MujocoGuidewireBackend:
        # 测试默认关闭碰撞,以免依赖 projects/cache/hulls
        cfg = MujocoGuidewireConfig(
            n_segments=cfg_overrides.pop("n_segments", 16),
            segment_length=cfg_overrides.pop("segment_length", 1.0),
            enable_collision=cfg_overrides.pop("enable_collision", False),
            **cfg_overrides,
        )
        return MujocoGuidewireBackend(
            entry_pos=[0.0, 0.0, 0.0],
            entry_dir=[0.0, 0.0, 1.0],
            config=cfg,
        )

    def joint_id(self, backend: MujocoGuidewireBackend, name: str) -> int:
        return int(
            backend._mj.mj_name2id(
                backend._model, backend._mj.mjtObj.mjOBJ_JOINT, name
            )
        )

    # ---- 基础结构 ----

    def test_capsule_chain_has_bend_slide_twist_joints(self):
        backend = self.make_backend(n_segments=8, segment_length=0.5)

        self.assertGreaterEqual(self.joint_id(backend, "prox_slide"), 0)
        self.assertGreaterEqual(self.joint_id(backend, "prox_twist"), 0)
        self.assertGreaterEqual(self.joint_id(backend, "bend_x_1"), 0)
        self.assertGreaterEqual(self.joint_id(backend, "bend_y_1"), 0)
        self.assertEqual(len(backend._site_ids), 8)

    def test_max_length_is_segments_times_segment_length(self):
        backend = self.make_backend(n_segments=20, segment_length=2.0)
        self.assertAlmostEqual(backend.max_length, 40.0)

    # ---- forward / backward ----

    def test_forward_backward_are_clamped(self):
        # 直丝(j_tip_angle_deg=0)隔离 J-tip 预弯:默认 65° 预弯下,wire XML 末端
        # 大角度回弯,刚推 0.25mm 的 tip site 会在 -Z 方向,违反"沿 entry_dir +Z"假设。
        # 本测试只验证 inserted_length clamp 行为,用直丝即可。
        backend = self.make_backend(n_segments=8, segment_length=0.5,
                                    j_tip_angle_deg=0.0)

        self.assertEqual(backend.get_guidewire_points().shape, (0, 3))

        self.assertTrue(backend.forward(0.25))
        self.assertAlmostEqual(backend.inserted_length, 0.25)
        pts = backend.get_guidewire_points()
        self.assertGreaterEqual(len(pts), 2)
        np.testing.assert_allclose(pts[0], np.array([0.0, 0.0, 0.0]), atol=1e-6)
        self.assertGreater(pts[-1, 2], pts[0, 2])

        self.assertTrue(backend.forward(1.0))
        self.assertAlmostEqual(backend.inserted_length, 1.25)

        self.assertTrue(backend.backward(0.5))
        self.assertAlmostEqual(backend.inserted_length, 0.75)

        self.assertTrue(backend.backward(100.0))
        self.assertAlmostEqual(backend.inserted_length, 0.0)
        self.assertEqual(backend.get_guidewire_points().shape, (0, 3))

    # ---- rotation ----

    def test_rotation_accumulates_independently_from_slide(self):
        # 用直丝(j_tip_angle_deg=0)隔离 twist 与 J-tip 预弯的耦合:
        # 默认 30° 预弯下,旋转 twist 会绕导丝主轴扫动 J 形 site 位置(这是预期物理),
        # 但本测试只想验证"twist 不会修改 inserted_length 也不会扰动直丝位置"。
        backend = self.make_backend(n_segments=8, segment_length=0.5,
                                    j_tip_angle_deg=0.0)
        backend.forward(1.0)
        before = backend.get_guidewire_points().copy()

        self.assertTrue(backend.rotate_left(math.pi / 2.0))
        self.assertAlmostEqual(backend.rotation_angle, math.pi / 2.0)
        self.assertAlmostEqual(backend.inserted_length, 1.0)
        np.testing.assert_allclose(
            backend.get_guidewire_points(), before, atol=1e-6,
        )

    # ---- config / rebuild ----

    def test_rebuild_changes_segment_count_and_preserves_inserted_length(self):
        backend = self.make_backend(n_segments=10, segment_length=1.0)
        backend.forward(3.0)
        self.assertAlmostEqual(backend.inserted_length, 3.0)

        new_cfg = backend.config.copy(n_segments=20, segment_length=1.0)
        backend.rebuild(new_cfg)

        self.assertEqual(backend.num_segments, 20)
        self.assertAlmostEqual(backend.max_length, 20.0)
        # inserted_length 应当被尽量保留(3 mm < 新 max 20 mm)
        self.assertAlmostEqual(backend.inserted_length, 3.0, places=4)

    def test_rebuild_with_different_radius_updates_geometry(self):
        backend = self.make_backend(n_segments=8, radius=0.2)
        self.assertAlmostEqual(backend.radius, 0.2)

        new_cfg = backend.config.copy(radius=0.4)
        backend.rebuild(new_cfg)
        self.assertAlmostEqual(backend.radius, 0.4)

    def test_stiffness_cascade_is_applied(self):
        # 间接验证:不同 decay 时近端段刚度应该一样,远端段应该不同
        b1 = self.make_backend(n_segments=8, stiffness=20.0, stiffness_decay=1.0)
        b2 = self.make_backend(n_segments=8, stiffness=20.0, stiffness_decay=0.9)
        # XML 里 bend_x_1 的 stiffness 应该都是 20*decay^1
        self.assertIn("stiffness=\"20.000000\"", b1._xml)
        self.assertIn("stiffness=\"18.000000\"", b2._xml)

    # ---- collision (skipped if no hulls) ----

    def test_loads_vessel_hulls_when_collision_enabled(self):
        hulls = Path(__file__).resolve().parents[1] / "projects" / "cache" / "hulls"
        if not hulls.exists() or not list(hulls.glob("hull_*.stl")):
            self.skipTest("projects/cache/hulls 未生成, 跳过血管碰撞测试")
        cfg = MujocoGuidewireConfig(
            n_segments=8, segment_length=1.0, enable_collision=True,
        )
        backend = MujocoGuidewireBackend(
            entry_pos=[0, 0, 0], entry_dir=[0, 0, 1], config=cfg,
        )
        self.assertGreater(backend.loaded_hull_count, 0)
        # 应该能调到 mj_step 不炸
        backend.forward(0.5)
        backend.step_simulation(3)


class MujocoGuidewireWrapperTest(unittest.TestCase):
    def test_wrapper_accepts_config_and_supports_rotation(self):
        centerline = Centerline(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            )
        )
        cfg = MujocoGuidewireConfig(
            n_segments=6, segment_length=0.5, enable_collision=False,
        )
        gw = MujocoGuidewire(centerline, config=cfg, push_step_mm=0.2)

        self.assertTrue(gw.supports_rotation())
        self.assertEqual(gw.backend.num_segments, 6)

        gw.advance_step()
        gw.rotate_clockwise()
        pts = gw.get_polyline_points()
        self.assertGreaterEqual(len(pts), 2)

    def test_apply_config_rebuilds_backend(self):
        centerline = Centerline(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        gw = MujocoGuidewire(
            centerline,
            config=MujocoGuidewireConfig(
                n_segments=6, segment_length=0.5, enable_collision=False,
            ),
        )
        gw.advance_step()
        before_len = gw.backend.inserted_length

        gw.apply_config(gw.config.copy(n_segments=12, segment_length=0.5))
        self.assertEqual(gw.backend.num_segments, 12)
        # 插入长度尽量保留
        self.assertAlmostEqual(gw.backend.inserted_length, before_len, places=4)

    def test_legacy_kwargs_still_work(self):
        # workspace_page 旧调用风格:radius/num_segments 直接传入
        centerline = Centerline(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        gw = MujocoGuidewire(
            centerline,
            radius=0.5,
            num_segments=8,
            segment_length=0.5,
            enable_contact=False,
        )
        self.assertAlmostEqual(gw.radius, 0.5)
        self.assertEqual(gw.backend.num_segments, 8)


if __name__ == "__main__":
    unittest.main()
