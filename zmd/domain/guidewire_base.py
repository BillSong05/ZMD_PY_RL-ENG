"""导丝抽象基类 + 动作枚举 — 为后续真实物理后端 / RL Agent 接入预留统一接口。

设计原则：
- UI / 仿真 / RL 都通过 GuidewireBase 与导丝交互
- 当前 MujocoGuidewire 支持前进/后退/旋转 + SDF 碰撞检测
- 真实物理后端 / RLAgentGuidewire 可后续无缝接入
- supports_rotation() 等能力声明使 UI 在不支持时静默
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class GuidewireAction(Enum):
    """统一动作枚举 — UI 键盘 / 按钮 / RL agent 都用此发送动作。"""
    ADVANCE = "advance"
    RETRACT = "retract"
    ROTATE_CW = "rotate_cw"
    ROTATE_CCW = "rotate_ccw"


class GuidewireBase(ABC):
    """导丝统一接口。

    实现类必须提供：
    - 运动：advance_step / retract_step / reset
    - 连续按住：start_hold_forward / start_hold_backward / stop_hold
    - 外观：set_radius / set_color
    - 查询：get_tip_point / get_tip_tangent / get_progress / get_distance_to_end

    可选（不支持时返回 False / 不实现）：
    - rotate_clockwise / rotate_counterclockwise
    - supports_rotation() 显式声明能力

    RL / 自动化通过 step(action) 统一调用。
    """

    type_name: str = "unknown"
    type_label: str = "Unnamed Guidewire"

    # ------------------------------------------------------------------
    # 运动 — 必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def advance_step(self) -> bool: ...

    @abstractmethod
    def retract_step(self) -> bool: ...

    @abstractmethod
    def reset(self) -> None: ...

    # ------------------------------------------------------------------
    # 连续按住 — 必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def start_hold_forward(self) -> None: ...

    @abstractmethod
    def start_hold_backward(self) -> None: ...

    @abstractmethod
    def stop_hold(self) -> None: ...

    # ------------------------------------------------------------------
    # 旋转 — 默认不支持
    # ------------------------------------------------------------------

    def rotate_clockwise(self) -> bool:
        return False

    def rotate_counterclockwise(self) -> bool:
        return False

    def supports_rotation(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # 外观 — 必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def set_radius(self, radius: float) -> None: ...

    def set_color(self, color: str) -> None:
        """默认 no-op；子类可自定义。"""
        pass

    # ------------------------------------------------------------------
    # 查询 — 必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def get_tip_point(self) -> Optional[np.ndarray]: ...

    @abstractmethod
    def get_tip_tangent(self) -> Optional[np.ndarray]: ...

    @abstractmethod
    def get_progress(self) -> float: ...

    @abstractmethod
    def get_distance_to_end(self) -> float: ...

    # ------------------------------------------------------------------
    # RL / 自动化预留 — 子类一般不必重写
    # ------------------------------------------------------------------

    def step(self, action: GuidewireAction) -> Dict[str, Any]:
        """统一动作入口。返回 dict，包含 action / success / observation。"""
        if action == GuidewireAction.ADVANCE:
            ok = self.advance_step()
        elif action == GuidewireAction.RETRACT:
            ok = self.retract_step()
        elif action == GuidewireAction.ROTATE_CW:
            ok = self.rotate_clockwise()
        elif action == GuidewireAction.ROTATE_CCW:
            ok = self.rotate_counterclockwise()
        else:
            ok = False
        info = {"action": action.value, "success": bool(ok)}
        info.update(self.get_observation())
        return info

    def get_observation(self) -> Dict[str, Any]:
        return {
            "tip": self.get_tip_point(),
            "tangent": self.get_tip_tangent(),
            "progress": self.get_progress(),
            "distance_to_end": self.get_distance_to_end(),
        }

    def get_state(self) -> Dict[str, Any]:
        """RL 训练用 — 默认返回 observation。真实物理导丝可加 collision/contact 等。"""
        return self.get_observation()
