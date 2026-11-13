"""手动路径构建器 — 根据用户提供的控制点生成中心线。"""
from typing import List

import numpy as np
from scipy.interpolate import interp1d, CubicSpline

from zmd.centerline.centerline_model import Centerline


def build_polyline_from_points(
    points: List[np.ndarray],
    source_point: np.ndarray = None,
    target_point: np.ndarray = None,
) -> Centerline:
    """从一组 3D 控制点构建折线中心线。

    Args:
        points: 控制点列表，每个元素是 (3,) 的 np.ndarray。
        source_point: 入口点（可选）。
        target_point: 出口点（可选）。

    Returns:
        Centerline 实例。
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("At least two 3D control points are required")

    return Centerline(
        points=pts,
        source_point=(
            np.asarray(source_point, dtype=float).flatten()[:3]
            if source_point is not None
            else pts[0].copy()
        ),
        target_point=(
            np.asarray(target_point, dtype=float).flatten()[:3]
            if target_point is not None
            else pts[-1].copy()
        ),
        source="manual",
    )


def build_cubic_spline_from_points(
    points: List[np.ndarray],
    num_samples: int = 80,
    source_point: np.ndarray = None,
    target_point: np.ndarray = None,
) -> Centerline:
    """从一组 3D 控制点构建平滑三次样条中心线。

    Args:
        points: 控制点列表。
        num_samples: 中心线采样点数。
        source_point: 入口点。
        target_point: 出口点。

    Returns:
        平滑后的 Centerline 实例。
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        # 控制点太少时回退到折线
        return build_polyline_from_points(points, source_point, target_point)

    param = np.linspace(0, 1, len(pts))
    param_fine = np.linspace(0, 1, num_samples)

    try:
        spline = CubicSpline(param, pts, bc_type="natural")
        sampled = spline(param_fine)
    except Exception:
        # 样条失败时回退到线性插值
        sampled = np.column_stack([
            np.interp(param_fine, param, pts[:, i]) for i in range(3)
        ])

    return Centerline(
        points=sampled,
        source_point=(
            np.asarray(source_point, dtype=float).flatten()[:3]
            if source_point is not None
            else sampled[0].copy()
        ),
        target_point=(
            np.asarray(target_point, dtype=float).flatten()[:3]
            if target_point is not None
            else sampled[-1].copy()
        ),
        source="manual",
    )
