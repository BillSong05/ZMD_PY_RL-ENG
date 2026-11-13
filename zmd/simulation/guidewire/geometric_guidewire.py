"""GeometricGuidewire — 临时几何导丝，纯几何 polyline + tip 圆弧。

⚠️ 不是真的 Cosserat Rod / 弹性杆 — 没有真实物理求解。
真实物理后端（PyBullet / 自研 Cosserat 离散段 等）待接入。

本类只做：
  - base 位置 + heading 方向（初始化时取中心线入口）
  - 直线 + tip 圆弧的 polyline
  - advance: 沿 heading 推进；retract: 撤回
  - rotate: 改变 tip 弯曲平面绕 heading 轴的角度
不再沿中心线 index 走 —— 没有血管碰撞约束。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import QObject, QTimer

from zmd.centerline.centerline_model import Centerline
from zmd.domain.guidewire_base import GuidewireAction, GuidewireBase
from zmd.domain.intervention_state import GuidewireState, InterventionState


HOLD_INTERVAL_MS = 90


class GeometricGuidewire(GuidewireBase):
    """临时几何导丝 — 不接触血管壁，自由直线 + tip 圆弧。"""

    type_name = "geometric"
    type_label = "Geometric J-Tip Guidewire"

    DEFAULT_RADIUS = 0.05
    MIN_LENGTH_MM = 1.5
    MAX_LENGTH_MM = 250.0
    STEP_LENGTH_MM = 1.0     # advance/retract 单步插入长度
    BEND_LENGTH = 2.4        # tip 弯曲段长度 (mm)
    BEND_ANGLE_DEG = 26      # tip 圆弧弯曲角度
    BEND_SEGMENTS = 10
    ROTATE_STEP_DEG = 15
    BASE_SEGMENT_MM = 0.6    # 直线段离散步长

    def __init__(self, centerline: Centerline,
                 on_state_change: Optional[Callable] = None,
                 radius: float = DEFAULT_RADIUS,
                 color: str = "#94a3b8"):
        # centerline 只用于初始化 base / heading / target，不作为运动轨迹约束
        self._centerline = centerline
        ref = np.asarray(centerline.points, dtype=float)
        if len(ref) < 2:
            raise ValueError("The centerline requires at least two points")
        self._base_pos = ref[0].copy()
        lookahead = min(6, len(ref) - 1)
        init_dir = ref[lookahead] - ref[0]
        n = float(np.linalg.norm(init_dir))
        if n > 1e-6:
            self._heading = init_dir / n
        else:
            self._heading = np.array([0.0, 1.0, 0.0])
        self._target = ref[-1].copy()
        self._initial_base_to_target = float(np.linalg.norm(self._target - self._base_pos))

        # 运动状态
        self._length = max(self.MIN_LENGTH_MM, 3.0)
        self._bend_phase_deg = 0.0

        # 外观
        self.radius = float(max(0.005, radius))
        self.color = str(color)
        self.on_state_change = on_state_change

        # 长按
        self._hold_owner = QObject()
        self._hold_timer = QTimer(self._hold_owner)
        self._hold_timer.setInterval(HOLD_INTERVAL_MS)
        self._hold_timer.timeout.connect(self._on_hold_tick)
        self._hold_action: Optional[GuidewireAction] = None

    # ------------------------------------------------------------------
    # 运动
    # ------------------------------------------------------------------

    def advance_step(self) -> bool:
        new_len = min(self.MAX_LENGTH_MM, self._length + self.STEP_LENGTH_MM)
        if abs(new_len - self._length) < 1e-6:
            return False
        self._length = new_len
        self._emit(); return True

    def retract_step(self) -> bool:
        new_len = max(self.MIN_LENGTH_MM, self._length - self.STEP_LENGTH_MM)
        if abs(new_len - self._length) < 1e-6:
            return False
        self._length = new_len
        self._emit(); return True

    def reset(self) -> None:
        self._length = max(self.MIN_LENGTH_MM, 3.0)
        self._bend_phase_deg = 0.0
        self._emit()

    # ------------------------------------------------------------------
    # 旋转 — 改弯曲平面
    # ------------------------------------------------------------------

    def supports_rotation(self) -> bool:
        return True

    def rotate_clockwise(self) -> bool:
        self._bend_phase_deg = (self._bend_phase_deg + self.ROTATE_STEP_DEG) % 360.0
        self._emit(); return True

    def rotate_counterclockwise(self) -> bool:
        self._bend_phase_deg = (self._bend_phase_deg - self.ROTATE_STEP_DEG) % 360.0
        self._emit(); return True

    # ------------------------------------------------------------------
    # 长按
    # ------------------------------------------------------------------

    def start_hold_forward(self) -> None:
        self._hold_action = GuidewireAction.ADVANCE
        self._on_hold_tick(); self._hold_timer.start()

    def start_hold_backward(self) -> None:
        self._hold_action = GuidewireAction.RETRACT
        self._on_hold_tick(); self._hold_timer.start()

    def stop_hold(self) -> None:
        self._hold_timer.stop(); self._hold_action = None

    def _on_hold_tick(self) -> None:
        if self._hold_action == GuidewireAction.ADVANCE:
            self.advance_step()
        elif self._hold_action == GuidewireAction.RETRACT:
            self.retract_step()

    # ------------------------------------------------------------------
    # 外观
    # ------------------------------------------------------------------

    def set_radius(self, radius: float) -> None:
        self.radius = float(max(0.005, radius))

    def set_color(self, color: str) -> None:
        self.color = str(color)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_polyline_points(self) -> np.ndarray:
        """直线段 (base → tip_start) + tip 弯曲段。"""
        t_hat = self._heading
        n_base = max(2, int(self._length / self.BASE_SEGMENT_MM) + 1)
        s = np.linspace(0.0, self._length, n_base).reshape(-1, 1)
        base_pts = self._base_pos[None, :] + s * t_hat[None, :]
        tip_start = base_pts[-1]

        # tip 弯曲基准
        world_up = np.array([0.0, 0.0, 1.0])
        e1 = world_up - float(np.dot(world_up, t_hat)) * t_hat
        e1n = float(np.linalg.norm(e1))
        if e1n < 1e-6:
            e1 = np.array([1.0, 0.0, 0.0])
            e1 = e1 - float(np.dot(e1, t_hat)) * t_hat
            e1n = float(np.linalg.norm(e1))
        e1 = e1 / e1n if e1n > 1e-6 else np.array([1.0, 0.0, 0.0])
        e2 = np.cross(t_hat, e1)
        phase = np.radians(self._bend_phase_deg)
        bend_dir = e1 * np.cos(phase) + e2 * np.sin(phase)

        bend_rad = np.radians(self.BEND_ANGLE_DEG)
        R = self.BEND_LENGTH / bend_rad if bend_rad > 1e-6 else self.BEND_LENGTH * 1e3
        bend_pts = np.empty((self.BEND_SEGMENTS, 3))
        for i in range(self.BEND_SEGMENTS):
            theta_i = bend_rad * (i + 1) / self.BEND_SEGMENTS
            bend_pts[i] = tip_start + R * np.sin(theta_i) * t_hat + R * (1.0 - np.cos(theta_i)) * bend_dir
        return np.vstack([base_pts, bend_pts])

    def get_tip_point(self) -> Optional[np.ndarray]:
        return self.get_polyline_points()[-1].copy()

    def get_tip_tangent(self) -> Optional[np.ndarray]:
        pts = self.get_polyline_points()
        if len(pts) < 2:
            return None
        v = pts[-1] - pts[-2]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else None

    def get_progress(self) -> float:
        """tip → 目标的距离倒推估算（非真正中心线进度，仅参考）。"""
        if self._initial_base_to_target < 1e-6:
            return 1.0
        d = float(np.linalg.norm(self._target - self.get_tip_point()))
        return float(max(0.0, min(1.0, 1.0 - d / self._initial_base_to_target)))

    def get_distance_to_end(self) -> float:
        return float(np.linalg.norm(self._target - self.get_tip_point()))

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    def get_intervention_state(self) -> InterventionState:
        polyline = self.get_polyline_points()
        return InterventionState(
            guidewire=GuidewireState(
                centerline=polyline.copy(),
                tip_position=polyline[-1].copy(),
                inserted_index=len(polyline) - 1,
                radius=self.radius,
            ),
            time=0.0, step_count=0,
        )

    def _emit(self) -> None:
        if self.on_state_change is None:
            return
        try:
            self.on_state_change(self.get_intervention_state())
        except Exception:
            pass

    @property
    def centerline(self) -> Centerline:
        return self._centerline
