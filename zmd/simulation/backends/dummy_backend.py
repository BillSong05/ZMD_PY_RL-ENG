"""演示用物理后端 — 沿 reference_centerline 推进导丝，不涉及真实物理。

不再随机生成导丝路径，所有推进沿给定的 reference_centerline 进行。
后续可替换为弹性杆模型驱动的推进逻辑。

action 格式：
    {"insert": N}   — 沿中心线前进 N 个索引点（正=前进，负=后退）
    {"reset": True} — 重置到入口
"""
from typing import Any, Dict, Optional
import numpy as np

from zmd.simulation.backends.base_backend import BasePhysicsBackend
from zmd.domain.intervention_state import InterventionState, GuidewireState
from zmd.centerline.centerline_model import Centerline


class DummyPhysicsBackend(BasePhysicsBackend):
    """沿 reference_centerline 逐点推进导丝的演示后端。"""

    MIN_VALID_INDEX = 2          # 至少 3 个可见点，保证 tangent / camera 计算
    SAFETY_BACK_MARGIN = 1       # 末端不到达倒数 N 个点，避免 tip 贴边界

    def __init__(self, reference_centerline: Optional[Centerline] = None):
        self._state = InterventionState()
        self._reference: Optional[Centerline] = reference_centerline
        self._inserted_idx: int = self.MIN_VALID_INDEX

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def reset(self, initial_state: Any = None) -> InterventionState:
        """重置导丝 — 不退到 path[0] 而是 path[MIN_VALID_INDEX]，避免落到血管壁/入口外。"""
        self._inserted_idx = self.MIN_VALID_INDEX
        self._state.time = 0.0
        self._state.step_count = 0
        self._state = self._build_state()
        return self._state

    def step(self, action: Dict[str, Any], dt: float = 0.01) -> InterventionState:
        """沿 reference_centerline 推进一步。

        action = {"insert": 2}   → 前进 2 个索引点
        action = {"insert": -1}  → 后退 1 个索引点
        action = {"reset": True} → 回到入口
        """
        if self._reference is None:
            return self._state

        total_pts = self._reference.n_points

        prev_idx = self._inserted_idx
        min_idx = self.MIN_VALID_INDEX
        max_idx = max(min_idx, total_pts - 1 - self.SAFETY_BACK_MARGIN)
        if action.get("reset"):
            self._inserted_idx = min_idx
        elif "insert" in action:
            delta = int(action["insert"])
            self._inserted_idx = max(min_idx, min(max_idx, self._inserted_idx + delta))

        # 已贴住起点/终点 — 不重建状态，避免 auto-repeat 在边界持续重渲
        if self._inserted_idx == prev_idx and not action.get("reset"):
            return None

        self._state.time += dt
        self._state.step_count += 1
        self._state = self._build_state()
        return self._state

    def get_state(self) -> InterventionState:
        return self._state

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _build_state(self) -> InterventionState:
        if self._reference is None:
            return self._state

        # 截取 reference_centerline 的前 N 个点 — 保证 ≥ MIN_VALID_INDEX+1 个点
        visible_idx = max(self.MIN_VALID_INDEX + 1, self._inserted_idx + 1)
        n_avail = self._reference.n_points
        visible_idx = min(visible_idx, n_avail)
        visible_pts = self._reference.points[:visible_idx].copy()
        tip = visible_pts[-1].copy()

        return InterventionState(
            guidewire=GuidewireState(
                centerline=visible_pts,
                tip_position=tip,
                inserted_index=self._inserted_idx,
                radius=0.2,
            ),
            time=self._state.time,
            step_count=self._state.step_count,
        )
