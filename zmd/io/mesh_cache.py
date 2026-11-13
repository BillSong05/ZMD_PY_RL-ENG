"""Mesh cache — 在 workspace 内共享 STL 解析结果，避免每个场景重新 pv.read。

vessel STL ~60MB / 596k 顶点，pv.read 单次 ~3-5s。4 个场景并行读 = 12-20s 主线程阻塞。
缓存策略：以 (path, normalized) 为 key，存归一化后的 PolyData。后续场景直接 .copy()
即可（共享几何数据 / 不重复 IO）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pyvista as pv


class MeshCache:
    """页面级 STL polydata 缓存。"""

    DEFAULT_NORMALIZED_LENGTH = 900.0

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, float], pv.PolyData] = {}

    def get_normalized(self, path: Path,
                       normalized_length: float = DEFAULT_NORMALIZED_LENGTH) -> pv.PolyData:
        """读取并归一化（居中 + 等比缩放到 length）；同 path/length 第二次以后只 copy。"""
        key = (str(path), float(normalized_length))
        cached = self._cache.get(key)
        if cached is None:
            mesh = pv.read(path).copy()
            mesh.translate((-mesh.center[0], -mesh.center[1], -mesh.center[2]),
                           inplace=True)
            if mesh.length > 0:
                mesh.scale(normalized_length / mesh.length, inplace=True)
            self._cache[key] = mesh
            cached = mesh
        # 直接返回缓存对象 — add_mesh 不会 in-place 改 PolyData，多场景共享同一份 VTK 数据
        return cached
