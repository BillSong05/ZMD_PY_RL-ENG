"""中心线数据结构 — 支持单条路径以及多分支树。"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv


@dataclass
class CenterlineBranch:
    """中心线树中的一条 polyline 分支。"""
    points: np.ndarray              # (N, 3)
    radius: Optional[np.ndarray] = None  # (N,)

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    @property
    def n_points(self) -> int:
        return len(self.points)


@dataclass
class Centerline:
    """血管中心线 — points 为导丝实际遵循的路径。"""
    points: np.ndarray                          # (N, 3)
    radius: Optional[np.ndarray] = None         # (N,)
    source_point: Optional[np.ndarray] = None
    target_point: Optional[np.ndarray] = None
    source: str = "manual"
    branches: List[CenterlineBranch] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def total_length(self) -> float:
        diffs = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    # ------------------------------------------------------------------
    # VMTK .vtp 加载
    # ------------------------------------------------------------------

    @classmethod
    def load_branches_from_vtp(cls, vtp_path: Path) -> List[CenterlineBranch]:
        """读取 VMTK 输出的 .vtp，把每条 polyline 解码为一个 CenterlineBranch。"""
        if not vtp_path.exists():
            return []
        mesh = pv.read(vtp_path)
        if mesh.n_points < 2:
            return []
        lines = mesh.lines
        if lines is None or len(lines) == 0:
            return [CenterlineBranch(points=np.asarray(mesh.points))]

        radius_arr = None
        for name in mesh.point_data.keys():
            if "Radius" in name or "radius" in name:
                radius_arr = np.asarray(mesh.point_data[name])
                break

        branches: List[CenterlineBranch] = []
        idx = 0
        flat = np.asarray(lines).ravel()
        while idx < len(flat):
            n = int(flat[idx])
            ids = flat[idx + 1 : idx + 1 + n].astype(int)
            pts = np.asarray(mesh.points)[ids]
            r = radius_arr[ids] if radius_arr is not None else None
            branches.append(CenterlineBranch(points=pts, radius=r))
            idx += 1 + n
        return branches

    @classmethod
    def load_from_vtp(
        cls,
        vtp_path: Path,
        source_point: Optional[np.ndarray] = None,
        target_point: Optional[np.ndarray] = None,
    ) -> Optional["Centerline"]:
        """从 VMTK .vtp 加载，并按 source/target 选出最匹配的路径。

        若不提供 source/target，则返回最长分支作为默认导丝路径，
        同时 branches 字段保留全部分支以便渲染整棵树。
        """
        branches = cls.load_branches_from_vtp(vtp_path)
        if not branches:
            return None

        if source_point is None or target_point is None:
            chosen = max(branches, key=lambda b: b.n_points)
        else:
            chosen = pick_route(branches, np.asarray(source_point), np.asarray(target_point))

        # 保证 chosen.start 距 source 更近（必要时翻转）
        if source_point is not None:
            sp = np.asarray(source_point)
            if np.linalg.norm(chosen.start - sp) > np.linalg.norm(chosen.end - sp):
                chosen = CenterlineBranch(
                    points=chosen.points[::-1].copy(),
                    radius=None if chosen.radius is None else chosen.radius[::-1].copy(),
                )

        return cls(
            points=chosen.points,
            radius=chosen.radius,
            source_point=chosen.start.copy() if source_point is None else np.asarray(source_point),
            target_point=chosen.end.copy() if target_point is None else np.asarray(target_point),
            source="vmtk",
            branches=branches,
        )


def pick_route(
    branches: List[CenterlineBranch],
    source: np.ndarray,
    target: np.ndarray,
) -> CenterlineBranch:
    """从分支集合中挑出端点最贴近 source/target 的那条。

    VMTK 的中心线树一般含若干条共享起点的分支；本函数对每条分支
    比较 (start, end) 与 (source, target) 任一顺序下的距离之和。
    """
    best: Optional[Tuple[float, CenterlineBranch]] = None
    for b in branches:
        d1 = np.linalg.norm(b.start - source) + np.linalg.norm(b.end - target)
        d2 = np.linalg.norm(b.end - source) + np.linalg.norm(b.start - target)
        score = min(d1, d2)
        if best is None or score < best[0]:
            best = (score, b)
    assert best is not None
    return best[1]
