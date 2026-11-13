"""介入操作整体状态。"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from zmd.domain.vessel import Vessel
from zmd.domain.robot import Robot


@dataclass
class GuidewireState:
    """导丝运行时状态。

    与 zmd/domain/guidewire.py 中的 Guidewire（配置描述）职责不同。
    """
    centerline: np.ndarray        # (N, 3) 当前可见的中心线坐标
    tip_position: np.ndarray      # (3,)  尖端位置
    inserted_index: int = 0       # 已插入到的中心线索引
    radius: float = 0.2           # 管径
    # J-tip 弯曲平面方向标记 — 让 twist 旋转视觉可见(圆管绕自身轴转看不出)
    # {"position": (3,), "tangent": (3,), "bend_dir": (3,), "radius": float}
    # 没有 J-tip 弯曲或导丝类型不支持时为 None。
    tip_marker_pose: Optional[Any] = None


@dataclass
class InterventionState:
    """聚合一次介入仿真的完整状态。"""
    guidewire: Optional[GuidewireState] = None
    vessel: Optional[Vessel] = None
    robot: Optional[Robot] = None
    time: float = 0.0
    step_count: int = 0
    is_colliding: bool = False
    blocked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
