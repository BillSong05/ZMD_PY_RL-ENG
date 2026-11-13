"""导丝模型领域对象（弹性杆模型占位）。"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Guidewire:
    """导丝模型抽象，后续由弹性杆模型接管。"""
    name: str = "guidewire"
    length: float = 0.0
    radius: float = 0.025
    num_nodes: int = 18
    segment_length: float = 0.18
    tip_prebend_segments: int = 4
    tip_prebend_curvature: float = 0.18
    damping: float = 0.96

    # 运行时状态占位
    points: List[Tuple[float, float, float]] = field(default_factory=list)
    tangents: List[Tuple[float, float, float]] = field(default_factory=list)
    normals: List[Tuple[float, float, float]] = field(default_factory=list)
    binormals: List[Tuple[float, float, float]] = field(default_factory=list)
    insertion_depth: float = 0.0
    axial_rotation: float = 0.0

    # 扩展槽
    config: Dict[str, Any] = field(default_factory=dict)
