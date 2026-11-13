"""血管模型领域对象。"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Vessel:
    """血管模型抽象，封装 STL 路径和几何信息。"""
    name: str
    stl_path: Path
    entry_points: List[Tuple[float, float, float]] = field(default_factory=list)
    exit_points: List[Tuple[float, float, float]] = field(default_factory=list)
    opacity: float = 0.5
