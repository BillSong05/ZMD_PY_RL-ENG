"""GuidewireBase 包装层:把 MuJoCo 医疗导丝接入 UI/血管/中心线管线。"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import QObject, QTimer

from zmd.centerline.centerline_model import Centerline
from zmd.domain.guidewire_base import GuidewireBase
from zmd.domain.intervention_state import GuidewireState, InterventionState
from zmd.simulation.backends.mujoco_guidewire_backend import MujocoGuidewireBackend
from zmd.simulation.guidewire.mujoco_config import MujocoGuidewireConfig


HOLD_INTERVAL_MS = 90
# 调试旋转可视化时打印 twist / bend_dir / tip_endpoint:set env ZMD_GW_DEBUG=1
_DEBUG_TWIST = os.environ.get("ZMD_GW_DEBUG", "") not in ("", "0", "false", "False")


class MujocoGuidewire(GuidewireBase):
    """MuJoCo 医疗导丝(capsule-chain + 凸分解血管碰撞 + 可调物理参数)。"""

    type_name = "mujoco"
    type_label = "MuJoCo Medical Guidewire"

    # 这些常量保留供面板初值用 —— 真实默认在 MujocoGuidewireConfig
    DEFAULT_RADIUS = MujocoGuidewireConfig.radius
    DEFAULT_PUSH_STEP_MM = 0.5
    DEFAULT_ROTATE_STEP_RAD = 0.1
    DEFAULT_NUM_SEGMENTS = MujocoGuidewireConfig.n_segments
    DEFAULT_SEGMENT_LENGTH = MujocoGuidewireConfig.segment_length
    DEFAULT_ACTION_SUBSTEPS = MujocoGuidewireConfig.action_substeps
    DEFAULT_MJ_SUBSTEPS = MujocoGuidewireConfig.mj_substeps_per_action

    def __init__(
        self,
        centerline: Centerline,
        on_state_change: Optional[Callable] = None,
        config: Optional[MujocoGuidewireConfig] = None,
        push_step_mm: float = DEFAULT_PUSH_STEP_MM,
        rotate_step_rad: float = DEFAULT_ROTATE_STEP_RAD,
        # 兼容旧 kwargs(workspace_page 旧调用 / 测试)
        radius: Optional[float] = None,
        num_segments: Optional[int] = None,
        segment_length: Optional[float] = None,
        action_substeps: Optional[int] = None,
        mj_substeps: Optional[int] = None,
        **_unused,
    ):
        self._centerline = centerline
        self._on_state_change = on_state_change

        cfg = config or MujocoGuidewireConfig()
        if radius is not None:
            cfg = cfg.copy(radius=float(radius))
        if num_segments is not None:
            cfg = cfg.copy(n_segments=int(num_segments))
        if segment_length is not None:
            cfg = cfg.copy(segment_length=float(segment_length))
        if action_substeps is not None:
            cfg = cfg.copy(action_substeps=int(action_substeps))
        if mj_substeps is not None:
            cfg = cfg.copy(mj_substeps_per_action=int(mj_substeps))
        self.config = cfg

        self.radius = float(max(0.005, cfg.radius))
        self.color = "#b8bcc7"
        self.push_step_mm = float(max(0.001, push_step_mm))
        self.rotate_step_rad = float(rotate_step_rad)

        points = np.asarray(centerline.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError(
                "MujocoGuidewire requires a centerline with at least 2 points"
            )

        entry_pos = points[0].copy()
        entry_dir = points[1] - points[0]
        direction_norm = float(np.linalg.norm(entry_dir))
        if direction_norm < 1e-9:
            entry_dir = np.array([1.0, 0.0, 0.0])
        else:
            entry_dir = entry_dir / direction_norm
        self._entry_pos = entry_pos
        self._entry_dir = entry_dir

        # 把 centerline 传给 backend → 启用 CenterlineTubeConstraint
        lumen_radius = getattr(centerline, "radius", None)
        self.backend = MujocoGuidewireBackend(
            entry_pos=entry_pos,
            entry_dir=entry_dir,
            config=cfg,
            centerline_points=points,
            lumen_radius=lumen_radius,
        )

        self._hold_owner = QObject()
        self._hold_timer = QTimer(self._hold_owner)
        self._hold_timer.setInterval(HOLD_INTERVAL_MS)
        self._hold_timer.timeout.connect(self._on_hold_tick)
        self._hold_action: Optional[str] = None
        # 最近一次 _overlay_j_tip 计算出的 J-tip 弯曲平面信息(供渲染器画非对称
        # marker,让"绕轴旋转"在主场景中肉眼可见 —— 圆截面绕自身轴转看不出)。
        self._last_tip_marker_pose: Optional[dict] = None

    @staticmethod
    def auto_radius_from_centerline(
        centerline,
        factor: float = 0.35,
        min_radius: float = 0.02,
        max_radius: float = 0.5,
    ) -> Optional[float]:
        """根据中心线最小局部 lumen 半径自动建议导丝半径。

        规则: guidewire_radius = factor × min(lumen_radius),clip 到 [min_radius, max_radius]。
        如果中心线没有 per-point 半径数据则返回 None,调用方应使用 config 默认。
        """
        radii = getattr(centerline, "radius", None)
        if radii is None:
            return None
        arr = np.asarray(radii, dtype=float).reshape(-1)
        if len(arr) == 0 or not np.isfinite(arr).any():
            return None
        min_lumen = float(np.nanmin(arr))
        return float(np.clip(min_lumen * factor, min_radius, max_radius))

    def apply_config(self, new_config: MujocoGuidewireConfig) -> None:
        """面板"应用并重建"按钮调用:用新 config 重建 backend。"""
        self.config = new_config
        self.radius = float(max(0.005, new_config.radius))
        self.backend.rebuild(new_config)
        self._emit()

    def advance_step(self) -> bool:
        ok = self.backend.forward(step_size=self.push_step_mm)
        self._emit()
        return ok

    def retract_step(self) -> bool:
        ok = self.backend.backward(step_size=self.push_step_mm)
        self._emit()
        return ok

    def advance_by(self, step_mm: float) -> bool:
        ok = self.backend.forward(step_size=max(0.001, abs(float(step_mm))))
        self._emit()
        return ok

    def retract_by(self, step_mm: float) -> bool:
        ok = self.backend.backward(step_size=max(0.001, abs(float(step_mm))))
        self._emit()
        return ok

    def rotate_clockwise(self) -> bool:
        ok = self.backend.rotate_right(angle_rad=self.rotate_step_rad)
        self._emit()
        self._debug_log_twist("cw")
        return ok

    def rotate_counterclockwise(self) -> bool:
        ok = self.backend.rotate_left(angle_rad=self.rotate_step_rad)
        self._emit()
        self._debug_log_twist("ccw")
        return ok

    def rotate_by(self, angle_rad: float) -> bool:
        angle = float(angle_rad)
        if angle >= 0.0:
            ok = self.backend.rotate_right(angle_rad=abs(angle))
            direction = "cw"
        else:
            ok = self.backend.rotate_left(angle_rad=abs(angle))
            direction = "ccw"
        self._emit()
        self._debug_log_twist(direction)
        return ok

    def _debug_log_twist(self, direction: str) -> None:
        """ZMD_GW_DEBUG=1 时打印每次旋转后的 twist/bend/tip,验证视觉旋转生效。"""
        if not _DEBUG_TWIST:
            return
        twist = float(self.backend.rotation_angle)
        pose = self._last_tip_marker_pose or {}
        bend = pose.get("bend_dir")
        pos = pose.get("position")
        try:
            twist_deg = float(np.rad2deg(twist))
            bend_str = (f"({bend[0]:+.3f},{bend[1]:+.3f},{bend[2]:+.3f})"
                        if bend is not None else "None")
            pos_str = (f"({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})"
                       if pos is not None else "None")
            print(f"[gw twist] {direction} twist={twist_deg:+.2f}° "
                  f"bend_dir={bend_str} tip={pos_str}")
        except Exception:
            pass

    def supports_rotation(self) -> bool:
        return True

    def reset(self) -> None:
        self.backend.reset()
        self._emit()

    def start_hold_forward(self) -> None:
        self._hold_action = "advance"
        self.advance_step()
        self._hold_timer.start()

    def start_hold_backward(self) -> None:
        self._hold_action = "retract"
        self.retract_step()
        self._hold_timer.start()

    def stop_hold(self) -> None:
        self._hold_timer.stop()
        self._hold_action = None

    def _on_hold_tick(self) -> None:
        if self._hold_action == "advance":
            self.advance_step()
        elif self._hold_action == "retract":
            self.retract_step()

    def get_inset_camera_pose(self):
        """返回内窥镜小窗相机位姿 dict(position/focal_point/up),未插入时 None。

        锚定在 backend 物理 tip + heading(_tip_path 末端,SDF 约束在腔内)。
        不用视觉 J-tip overlay 做锚 —— 那个 overlay 在 65° 大弯下会偏出 lumen,
        让相机跟着跑到血管外。
        """
        try:
            return self.backend.compute_inset_camera_pose()
        except Exception:
            return None

    def set_collision_sdf(self, sdf_fn) -> None:
        """注入血管有向距离场函数,接入 SDF 硬安全检查。

        sdf_fn(pos) -> float:负值=腔内(安全),正值=腔外(穿模)。
        由 workspace_page 在切换/重建 MuJoCo 导丝后调用。
        传入 None 可关闭 SDF 碰撞检查。
        """
        self.backend.set_collision_sdf(sdf_fn)

    def set_radius(self, radius: float) -> None:
        # 热改半径只改渲染粗细;真实碰撞半径需要 rebuild 才生效
        self.radius = float(max(0.005, radius))
        self.backend.radius = self.radius

    def set_color(self, color: str) -> None:
        del color

    def get_polyline_points(self) -> np.ndarray:
        pts = self.backend.get_guidewire_points()
        return self._overlay_j_tip(pts)

    # ------------------------------------------------------------------
    # 可视 J-tip 叠加 — 把远端最后一段渲染成绕轴旋转的圆弧
    # ------------------------------------------------------------------
    # 物理 _tip_path 只记 3D 位置,不带 twist 信息,所以纯按弧长重建的可视
    # 折线在原地旋转时看不出任何变化("旋转不了"的视觉表现)。这里在远端
    # 叠加一段 J 形圆弧:弧长 = j_tip_segments × segment_length,弯曲角 =
    # j_tip_angle_deg,弯曲方向 = world-up × tangent 绕 tangent 旋转 twist
    # 角度。Chaikin 切角 / 渲染端的平滑保持不变。
    #
    # 不修改物理 tip_path / get_tip_position(它们仍代表"未弯曲"基准点),
    # 物理走向用 backend 内 J-tip steering(已经按 twist 偏转 move_dir)。
    # 视觉/物理双轨,转向的物理效果由 J-tip steering 体现,视觉效果由这里体现。

    def _overlay_j_tip(self, pts: np.ndarray) -> np.ndarray:
        # 默认无 pose(直丝/不可见);_apply_overlay 才会写入有效 pose。
        self._last_tip_marker_pose = None
        if pts is None or len(pts) < 2:
            return pts
        cfg = self.config
        angle_deg = float(getattr(cfg, "j_tip_angle_deg", 0.0))
        if abs(angle_deg) < float(getattr(cfg, "j_tip_min_angle_deg", 1.0)):
            return pts
        n_seg = int(max(1, getattr(cfg, "j_tip_segments", 1)))
        seg_len = float(getattr(cfg, "segment_length", 1.0))
        j_arclen = float(n_seg) * seg_len
        if j_arclen <= 1e-6:
            return pts
        total = self._polyline_arclength(pts)
        # 需要至少留一小段直丝,否则 J-tip 几乎贴入口、计算切向不稳
        min_shaft = max(0.5, seg_len * 0.5)
        if total <= j_arclen + min_shaft:
            return pts

        base, t_hat = self._sample_polyline_with_tangent(pts, total - j_arclen)
        # 弯曲参考方向(twist=0 时):世界 Z × tangent;退化时用世界 Y × tangent
        ref = np.cross(t_hat, np.array([0.0, 0.0, 1.0]))
        rn = float(np.linalg.norm(ref))
        if rn < 1e-9:
            ref = np.cross(t_hat, np.array([0.0, 1.0, 0.0]))
            rn = float(np.linalg.norm(ref))
        if rn < 1e-9:
            return pts
        ref = ref / rn

        twist = float(self.backend.rotation_angle)
        bend_dir = self._rotate_around_axis(ref, t_hat, twist)
        bn = float(np.linalg.norm(bend_dir))
        if bn < 1e-9:
            return pts
        bend_dir = bend_dir / bn

        angle_rad = float(np.deg2rad(angle_deg))
        radius = j_arclen / angle_rad
        K = max(8, int(j_arclen / 0.3))
        thetas = np.linspace(0.0, angle_rad, K)
        arc = (base[None, :]
               + (radius * np.sin(thetas))[:, None] * t_hat[None, :]
               + (radius * (1.0 - np.cos(thetas)))[:, None] * bend_dir[None, :])

        shaft = self._truncate_polyline_at_arclength(pts, total - j_arclen)
        merged = arc if len(shaft) == 0 else np.vstack([shaft, arc[1:]])

        # ---- 缓存非对称 marker pose:画在 J-tip 末端,标记弯曲平面 ----
        # 关键:bend_dir 随 twist 旋转,marker 围着导丝主轴公转,可视化 twist。
        tip_end = arc[-1]
        # 末端切向:对最后一段弧点求差,稳定指出 J-tip 朝向
        tip_tan = arc[-1] - arc[-2]
        tn = float(np.linalg.norm(tip_tan))
        tip_tan = tip_tan / tn if tn > 1e-9 else t_hat
        self._last_tip_marker_pose = {
            "position": np.asarray(tip_end, dtype=float).copy(),
            "tangent": np.asarray(tip_tan, dtype=float).copy(),
            "bend_dir": np.asarray(bend_dir, dtype=float).copy(),
            "radius": float(self.radius),
            "twist_angle": twist,
        }
        return merged

    @staticmethod
    def _polyline_arclength(pts: np.ndarray) -> float:
        if len(pts) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    @staticmethod
    def _sample_polyline_with_tangent(pts: np.ndarray, s: float):
        """在弧长 s 处采样点 + 局部切向(s 段两端单位差)。s clamp 到 [0, total]。"""
        pts = np.asarray(pts, dtype=float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(seg.sum())
        if total < 1e-9:
            return pts[0].copy(), np.array([1.0, 0.0, 0.0])
        s = float(min(max(s, 0.0), total))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        i = int(np.searchsorted(cum, s, side="right") - 1)
        i = max(0, min(i, len(seg) - 1))
        L = float(seg[i]) if seg[i] > 1e-9 else 1e-9
        t = (s - float(cum[i])) / L
        t = min(max(t, 0.0), 1.0)
        pt = pts[i] + t * (pts[i + 1] - pts[i])
        tan = (pts[i + 1] - pts[i])
        tn = float(np.linalg.norm(tan))
        tan = tan / tn if tn > 1e-9 else np.array([1.0, 0.0, 0.0])
        return pt, tan

    @staticmethod
    def _truncate_polyline_at_arclength(pts: np.ndarray, s: float) -> np.ndarray:
        """返回 pts 的前缀,使总弧长 = s(最后一点为弧长 s 处的精确插值)。"""
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2 or s <= 0.0:
            return pts[:1].copy() if len(pts) else pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(seg.sum())
        if s >= total:
            return pts.copy()
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        i = int(np.searchsorted(cum, s, side="right") - 1)
        i = max(0, min(i, len(seg) - 1))
        L = float(seg[i]) if seg[i] > 1e-9 else 1e-9
        t = (s - float(cum[i])) / L
        t = min(max(t, 0.0), 1.0)
        last = pts[i] + t * (pts[i + 1] - pts[i])
        return np.vstack([pts[: i + 1], last[None, :]])

    @staticmethod
    def _rotate_around_axis(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues:把 v 绕单位向量 axis 旋转 angle 弧度(右手)。"""
        an = float(np.linalg.norm(axis))
        if an < 1e-9:
            return np.asarray(v, dtype=float).copy()
        k = np.asarray(axis, dtype=float) / an
        v = np.asarray(v, dtype=float)
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)

    def get_tip_point(self) -> Optional[np.ndarray]:
        if self.backend.inserted_length <= 1e-9:
            return None
        return self.backend.get_tip_position()

    def get_tip_tangent(self) -> Optional[np.ndarray]:
        if self.backend.inserted_length <= 1e-9:
            return None
        return self.backend.get_tip_direction()

    def get_progress(self) -> float:
        # 弱中心线模式:进度基于 tip 投影到中心线的弧长,而不是 inserted_length。
        # 导丝偏离中心线后走过的实际路径可能 >= travel,但 tip 在中心线投影上
        # 可能还没到终点 —— 用 inserted/travel 会过早显示 100%。
        travel_length = self.backend.travel_length
        if travel_length <= 1e-9:
            return 0.0
        try:
            tip_s = self.backend.tip_centerline_s()
        except Exception:
            tip_s = self.backend.inserted_length
        return float(min(1.0, max(0.0, tip_s / travel_length)))

    def get_distance_to_end(self) -> float:
        try:
            tip_s = self.backend.tip_centerline_s()
        except Exception:
            tip_s = self.backend.inserted_length
        return float(max(0.0, self.backend.travel_length - tip_s))

    @property
    def centerline(self) -> Centerline:
        return self._centerline

    def get_intervention_state(self) -> InterventionState:
        # 用 backend.get_guidewire_points() 取物理 tip_path 投影 — 不应用视觉
        # J-tip overlay。J-tip 65° + 10mm 弧的横向偏移会让 overlay 末端跑出血管
        # lumen,渲染层"视觉穿模"。物理 SDF 没问题(检查的是 _tip_path,不是 overlay)。
        polyline = self.backend.get_guidewire_points()
        if len(polyline) == 0:
            return InterventionState(
                guidewire=GuidewireState(
                    centerline=np.empty((0, 3)),
                    tip_position=self.backend.entry_pos.copy(),
                    inserted_index=0,
                    radius=self.radius,
                ),
                time=0.0,
                step_count=0,
            )

        centerline_count = int(self._centerline.n_points)
        inserted_index = int(self.get_progress() * max(0, centerline_count - 1))
        return InterventionState(
            guidewire=GuidewireState(
                centerline=polyline.copy(),
                tip_position=polyline[-1].copy(),
                inserted_index=inserted_index,
                radius=self.radius,
                tip_marker_pose=self._last_tip_marker_pose,
            ),
            time=0.0,
            step_count=0,
        )

    def _emit(self) -> None:
        if self._on_state_change is None:
            return
        try:
            self._on_state_change(self.get_intervention_state())
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._hold_timer.stop()
        except Exception:
            pass
        try:
            self.backend.close()
        except Exception:
            pass
