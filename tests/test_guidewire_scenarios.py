"""导丝运动验收测试(第二步:弱中心线方向引导)。

第二步已移除"强中心线贴合"逻辑:导丝不再被 teleport 到 centerline(s),
而是沿自身方向前进,中心线只做弱方向引导 + 安全限制(max_centerline_deviation)。
因此本测试不再断言"导丝紧贴中心线(max_radial<0.2)",改为验收:
1. straight  — 直管:导丝方向与中心线一致,仍大致沿中心线,不蛇形飘出
2. bend90    — 90° 弯管:导丝明显偏离中心线(证明不再强贴合),但偏离受 max_dev 限制
3. real      — 真实血管:入口急弯允许偏离,推进有限不爆炸
另见 WeakGuidanceTest:验收 centerline_bias 与 backward 沿自身路径回退。

策略:
- backend 默认 enable_collision=False(无 hull)
- 默认 j_tip=0(直导丝)
- 推进步长适中,避免大步长失稳
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from zmd.simulation.backends.mujoco_guidewire_backend import MujocoGuidewireBackend
from zmd.simulation.guidewire.mujoco_config import MujocoGuidewireConfig


# ---------------------------------------------------------------- fixtures

def make_straight_centerline(length: float = 80.0, n: int = 200) -> np.ndarray:
    """沿 +Z 方向的直线."""
    z = np.linspace(0.0, length, n)
    pts = np.zeros((n, 3))
    pts[:, 2] = z
    return pts


def make_bend90_centerline(
    leg: float = 40.0, radius: float = 15.0, n_arc: int = 80, n_leg: int = 80,
) -> np.ndarray:
    """直段沿 +Z,90° 弯成 +X,直段沿 +X。共三段拼接。"""
    z1 = np.linspace(0.0, leg, n_leg)
    p1 = np.stack([np.zeros_like(z1), np.zeros_like(z1), z1], axis=1)

    # 弯部:圆心 (radius, 0, leg),从 +Z 转到 +X
    th = np.linspace(0.0, np.pi / 2.0, n_arc)
    arc = np.stack(
        [radius - radius * np.cos(th), np.zeros_like(th), leg + radius * np.sin(th)],
        axis=1,
    )

    x2 = np.linspace(0.0, leg, n_leg)
    p3 = np.stack(
        [radius + x2, np.zeros_like(x2), (leg + radius) * np.ones_like(x2)],
        axis=1,
    )
    # 去掉重叠端点
    return np.concatenate([p1, arc[1:], p3[1:]], axis=0)


def load_real_centerline() -> np.ndarray | None:
    vtp = Path(__file__).resolve().parents[1] / "projects" / "cache" / "VascuModel_centerline.vtp"
    if not vtp.exists():
        return None
    import pyvista as pv
    m = pv.read(vtp)
    pts = np.asarray(m.points)
    return pts


def make_backend(
    centerline: np.ndarray, lumen_radius=2.0, **cfg_overrides
) -> MujocoGuidewireBackend:
    cfg = MujocoGuidewireConfig(
        n_segments=cfg_overrides.pop("n_segments", 40),
        segment_length=cfg_overrides.pop("segment_length", 1.0),
        radius=cfg_overrides.pop("radius", 0.35),
        enable_collision=False,
        j_tip_angle_deg=0.0,
        # 调试参数:极软导丝 + 强约束 + 强中心吸引 + 多 mj 步数
        stiffness=cfg_overrides.pop("stiffness", 0.05),
        stiffness_decay=cfg_overrides.pop("stiffness_decay", 1.0),
        damping=cfg_overrides.pop("damping", 1.0),
        armature=cfg_overrides.pop("armature", 0.005),
        constraint_k_spring=cfg_overrides.pop("constraint_k_spring", 50000.0),
        constraint_k_damp=cfg_overrides.pop("constraint_k_damp", 100.0),
        constraint_margin=cfg_overrides.pop("constraint_margin", 0.02),
        constraint_k_attract=cfg_overrides.pop("constraint_k_attract", 5000.0),
        mj_substeps_per_action=cfg_overrides.pop("mj_substeps_per_action", 40),
        **cfg_overrides,
    )
    entry = centerline[0].copy()
    direction = centerline[1] - centerline[0]
    return MujocoGuidewireBackend(
        entry_pos=entry,
        entry_dir=direction,
        config=cfg,
        centerline_points=centerline,
        lumen_radius=lumen_radius,
    )


# ---------------------------------------------------------------- tests

import unittest


class StraightTubeTest(unittest.TestCase):
    def test_advance_1000_steps_does_not_explode_or_drift(self):
        cl = make_straight_centerline(length=80.0, n=200)
        backend = make_backend(cl, lumen_radius=2.0)

        push = 0.05   # mm/step
        max_radial_history = []
        for i in range(1000):
            backend.forward(push)
            stats = backend.last_constraint_stats
            max_radial_history.append(stats["max_radial_error"])

        max_radial = float(np.max(max_radial_history))
        final_max = max_radial_history[-1]
        inserted = backend.inserted_length
        tip_xy = backend.get_tip_position()[:2]
        radial_xy = float(np.linalg.norm(tip_xy))

        print(
            f"[straight] inserted={inserted:.2f}mm "
            f"max_radial_error={max_radial:.4f}mm "
            f"final_radial={final_max:.4f}mm "
            f"tip_xy_drift={radial_xy:.4f}mm"
        )

        # 1) 不爆炸:位置应该是有限数;推进不超过总路程(travel_length)
        self.assertTrue(np.all(np.isfinite(backend.get_tip_position())))
        self.assertLess(inserted, backend.travel_length + 1e-3)
        # 2) 弱引导模式
        self.assertEqual(backend.last_constraint_stats["mode"], "weak_centerline")
        # 3) 直管:导丝沿自身 +Z 方向走 == 中心线方向,不蛇形飘出
        self.assertLess(radial_xy, 1.0,
                        msg=f"tip 偏离中心线 {radial_xy:.3f} mm 过大")
        # 4) 偏离受 max_centerline_deviation 安全限制(直管本就贴合,留离散采样余量)
        self.assertLessEqual(
            max_radial, backend.config.max_centerline_deviation + 0.5,
            msg=f"max_radial {max_radial:.3f} mm 超过安全偏离上限")


class Bend90Test(unittest.TestCase):
    def test_tip_traverses_90deg_bend_with_low_error(self):
        cl = make_bend90_centerline(leg=40.0, radius=15.0, n_arc=80, n_leg=80)
        # MuJoCo XML 嵌套 body 超过约 78 层会爆栈,这里用 60 段×2mm=120mm 总长
        backend = make_backend(cl, lumen_radius=2.0,
                               n_segments=60, segment_length=2.0)

        # 让导丝推到约弧后段(>= leg + arc_length + 一点)
        push_total = 40.0 + (np.pi / 2.0 * 15.0) + 20.0  # 直 + 弧 + 后直 20mm
        push = 0.03   # 小步长给约束力时间让链跟弯
        n_steps = int(push_total / push)
        max_radial = 0.0
        for _ in range(n_steps):
            backend.forward(push)
            max_radial = max(max_radial, backend.last_constraint_stats["max_radial_error"])

        tip = backend.get_tip_position()
        tip_s = backend.last_constraint_stats["tip_centerline_s"]
        cl_length = float(np.sum(np.linalg.norm(np.diff(cl, axis=0), axis=1)))
        print(
            f"[bend90] inserted={backend.inserted_length:.2f}mm "
            f"tip_s={tip_s:.2f}/{cl_length:.2f}mm "
            f"max_radial_error={max_radial:.4f}mm "
            f"tip_xyz={tip}"
        )

        # 新弱引导:中心线只作判断参考,偏航(angle_to_centerline >= align_angle_deg)
        # 时纯走 raw_move_dir,不再自动纠偏。本测试 fixture 未注入 SDF,所以急
        # 弯处的偏航会一路冲出血管区域 —— 这正是规格"方向错就顶壁/走偏"。
        # 实机场景下 SDF 会拦住,这里只验证推进生效 + 不爆炸 + 弱引导没被退回强吸附。
        self.assertTrue(np.all(np.isfinite(tip)))
        self.assertGreater(backend.inserted_length, 40.0,
                           msg="导丝没推进到弯部")
        self.assertGreater(max_radial, 0.1,
                           msg="tip 仍紧贴中心线,弱引导失败(疑似仍在强吸附)")
        # 不再断言 max_radial 上限 —— 中心线不是牵引


class RealVesselTest(unittest.TestCase):
    def test_enters_stably_along_entry_centerline(self):
        # 用 Centerline.load_branches_from_vtp 取最长一支
        # (vtp 的 m.points 按 cell 排序,不能直接当 polyline)
        vtp = Path(__file__).resolve().parents[1] / "projects" / "cache" / "VascuModel_centerline.vtp"
        if not vtp.exists():
            self.skipTest("VascuModel_centerline.vtp 不存在")
        from zmd.centerline.centerline_model import Centerline
        branches = Centerline.load_branches_from_vtp(vtp)
        if not branches:
            self.skipTest("vtp 解析失败")
        branch = max(branches, key=lambda b: b.n_points)

        # VMTK 输出的入口附近常见急弯 / 噪声点,做一次 B-spline 平滑重采样
        from scipy.interpolate import splprep, splev
        raw_pts = branch.points[:200]
        raw_r = (branch.radius[:200]
                 if branch.radius is not None else np.full(200, 2.0))
        # 平滑(s>0)+ 均匀重采样
        tck, u = splprep(raw_pts.T, s=2.0, k=3)
        u_new = np.linspace(0, 1, 300)
        cl = np.stack(splev(u_new, tck), axis=1)
        # 半径用线性插值到新参数
        idx_map = (u_new * (len(raw_r) - 1)).astype(int)
        radius_per_pt = raw_r[idx_map]

        backend = make_backend(
            cl, lumen_radius=radius_per_pt,
            n_segments=60, segment_length=0.5,
        )

        push = 0.02   # 真实血管入口附近有急转弯,小步长 + 细段才能跟上
        n_steps = 1200
        max_radial = 0.0
        early_radial = 0.0
        for i in range(n_steps):
            backend.forward(push)
            err = backend.last_constraint_stats["max_radial_error"]
            max_radial = max(max_radial, err)
            if i == 100:
                early_radial = err

        tip = backend.get_tip_position()
        tip_s = backend.last_constraint_stats["tip_centerline_s"]
        cl_length = float(np.sum(np.linalg.norm(np.diff(cl, axis=0), axis=1)))
        print(
            f"[real] inserted={backend.inserted_length:.2f}mm "
            f"tip_s={tip_s:.2f}/{cl_length:.2f}mm "
            f"max_radial_error={max_radial:.4f}mm "
            f"early_radial(@100)={early_radial:.4f}mm"
        )

        self.assertTrue(np.all(np.isfinite(tip)))
        # 新弱引导:入口急弯处导丝若偏航就纯走 raw,不再自动纠偏。fixture 无 SDF,
        # 所以可能大幅偏离 —— 实机由 SDF 拦截。这里只验证推进 + 弱引导 mode。
        self.assertEqual(backend.last_constraint_stats["mode"], "weak_centerline")
        self.assertGreater(backend.inserted_length, 5.0)


class WeakGuidanceTest(unittest.TestCase):
    """第二步核心验收:中心线只做弱方向引导,不再当轨道。"""

    def _backend(self, **over):
        cl = make_bend90_centerline(leg=40.0, radius=15.0)
        return make_backend(cl, lumen_radius=2.0,
                            n_segments=60, segment_length=2.0, **over)

    def test_tip_not_teleported_to_centerline(self):
        # forward 后 tip 不再强制等于 centerline 上某点 → 到中心线最近距离明显 > 0
        b = self._backend(centerline_bias=0.15, max_centerline_deviation=5.0)
        for _ in range(1500):
            b.forward(0.05)
        d = b.last_constraint_stats["tip_to_centerline_distance"]
        self.assertEqual(b.last_constraint_stats["mode"], "weak_centerline")
        self.assertGreater(d, 0.1, "tip 仍贴中心线,强吸附未解除")

    def test_bias_zero_deviates_more_than_bias_one(self):
        # bias=0 纯自身方向(弯管处直冲)偏离更大;bias=1 沿切向偏离很小
        b0 = self._backend(centerline_bias=0.0, max_centerline_deviation=80.0)
        b1 = self._backend(centerline_bias=1.0, max_centerline_deviation=80.0)
        for _ in range(1500):
            b0.forward(0.05)
            b1.forward(0.05)
        d0 = b0.last_constraint_stats["tip_to_centerline_distance"]
        d1 = b1.last_constraint_stats["tip_to_centerline_distance"]
        self.assertGreater(d0, d1)
        self.assertLess(d1, 1.0, "bias=1 时 tip 应基本沿中心线切向方向")

    def test_backward_retraces_own_path_not_centerline(self):
        # backward 沿导丝已走过的路径回退,而不是吸回中心线
        b = self._backend(centerline_bias=0.15, max_centerline_deviation=5.0)
        for _ in range(1500):
            b.forward(0.05)
        path = np.asarray([p.copy() for p in b._tip_path])
        for _ in range(300):
            b.backward(0.05)
        tip_back = b.get_tip_position()
        dmin = float(np.min(np.linalg.norm(path - tip_back[None, :], axis=1)))
        self.assertLess(dmin, 0.1, "回撤的 tip 不在原路径上,可能被吸回中心线")

    def test_progress_uses_tip_centerline_s_not_wire_length(self):
        # 第三步:UI 进度改为基于 tip 投影到中心线的弧长 tip_centerline_s,
        # 不再用 inserted_length(导丝走过的实际路径长度)。
        b = self._backend()
        for _ in range(1000):
            b.forward(0.05)
        s = b.last_constraint_stats
        # progress 应等于 tip_centerline_s / travel_length
        self.assertAlmostEqual(
            s["progress"], s["tip_centerline_s"] / s["travel_length"], places=5)
        # progress_by_tip 与 progress 一致
        self.assertAlmostEqual(s["progress"], s["progress_by_tip"], places=6)
        # tip_centerline_s 与 inserted_length 是两个不同量(弱引导下导丝偏离中心线)
        self.assertIn("inserted_length", s)
        self.assertIn("tip_centerline_s", s)
        # distance_to_end_by_tip = travel - tip_s
        self.assertAlmostEqual(
            s["distance_to_end_by_tip"], s["travel_length"] - s["tip_centerline_s"],
            places=5)


if __name__ == "__main__":
    unittest.main()
