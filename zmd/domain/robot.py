"""机器人模型领域对象。"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Robot:
    """机器人/导管模型抽象。"""
    name: str
    stl_path: Path
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 0.1
