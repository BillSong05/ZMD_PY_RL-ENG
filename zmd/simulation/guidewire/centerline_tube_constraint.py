"""中心线管腔约束:对每个 guidewire body 施径向回拉弹簧力,使其不离开管腔。

实现方式
--------
1. 预存中心线 polyline (N,3) + 每点弧长 s + 每点局部 lumen radius (N,)
2. 每步:
   - 用 KD-tree 找每个 body 最近的中心线点(更精确做法可投影到段上,
     但点稠密时段近似==点近似)
   - 计算 radial_vector = body_pos - nearest_centerline_point
   - allowed = lumen_radius - guidewire_radius - margin
   - 若 radial_dist > allowed → 在 xfrc_applied[body, :3] 加径向回拉弹簧力
     F = -k_spring * (radial_dist - allowed) * radial_unit - k_damp * v_radial
3. 返回诊断:max_radial_error / num_projected_bodies / tip_centerline_s

不替代真实碰撞,但能在没有 STL hull 的情况下让导丝沿管腔推进,
对调试导丝 PD 稳定性 / 中心线接近策略足够。
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np


class CenterlineTubeConstraint:
    def __init__(
        self,
        centerline_points: np.ndarray,
        lumen_radius: Union[float, np.ndarray],
        guidewire_radius: float,
        k_spring: float = 5000.0,
        k_damp: float = 20.0,
        margin: float = 0.05,
        k_attract: float = 0.0,
    ):
        pts = np.asarray(centerline_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise ValueError("centerline_points must have shape (N>=2, 3)")
        self.points = pts.copy()

        # 弧长(从起点)
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])

        if np.ndim(lumen_radius) == 0:
            self.lumen_radius = np.full(len(self.points), float(lumen_radius))
        else:
            arr = np.asarray(lumen_radius, dtype=float).reshape(-1)
            if len(arr) != len(self.points):
                raise ValueError(
                    f"lumen_radius length {len(arr)} does not match "
                    f"centerline length {len(self.points)}"
                )
            self.lumen_radius = arr.copy()

        self.guidewire_radius = float(guidewire_radius)
        self.k_spring = float(k_spring)
        self.k_damp = float(k_damp)
        self.margin = float(margin)
        self.k_attract = float(k_attract)

        # KD-tree(scipy 已是项目依赖)
        try:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.points)
        except Exception:
            self._tree = None

    # ------------------------------------------------------------------

    def _closest_idx(self, points: np.ndarray) -> np.ndarray:
        if self._tree is not None:
            _, idxs = self._tree.query(points, k=1)
            return np.atleast_1d(idxs).astype(int)
        diff = points[:, None, :] - self.points[None, :, :]
        d2 = (diff * diff).sum(-1)
        return np.argmin(d2, axis=1)

    def step(self, model, data, body_ids: list[int]) -> dict:
        """对 body_ids 列表中的每个 body 计算 radial overshoot 并施力。

        返回:
            {"max_radial_error": mm, "num_projected_bodies": int, "tip_centerline_s": mm}
        """
        if not body_ids:
            return {
                "max_radial_error": 0.0,
                "num_projected_bodies": 0,
                "tip_centerline_s": 0.0,
            }

        body_ids_arr = np.asarray(body_ids, dtype=int)
        body_pos = np.asarray(data.xpos[body_ids_arr], dtype=float)   # (B, 3)
        # cvel = (angular, linear),取线速度后 3 维
        body_lin_vel = np.asarray(data.cvel[body_ids_arr, 3:6], dtype=float)

        idxs = self._closest_idx(body_pos)
        nearest = self.points[idxs]                                    # (B, 3)
        radial_vec = body_pos - nearest                                # (B, 3)
        radial_dist = np.linalg.norm(radial_vec, axis=1)               # (B,)

        local_lumen = self.lumen_radius[idxs]                          # (B,)
        allowed = np.maximum(local_lumen - self.guidewire_radius - self.margin, 0.0)
        overshoot = radial_dist - allowed                              # (B,)
        pos_overshoot = np.maximum(overshoot, 0.0)

        max_err = float(pos_overshoot.max())
        num_proj = int((pos_overshoot > 0).sum())

        mask = pos_overshoot > 0
        if mask.any():
            radial_unit = np.zeros_like(radial_vec)
            valid_dist = radial_dist[mask]
            valid_dist = np.maximum(valid_dist, 1e-9)
            radial_unit[mask] = radial_vec[mask] / valid_dist[:, None]

            # F_spring = -k * overshoot * unit
            force = -self.k_spring * pos_overshoot[:, None] * radial_unit
            # F_damp: 沿径向的速度分量(只阻尼往外那部分)
            v_rad_scalar = (body_lin_vel * radial_unit).sum(axis=1)
            v_rad_scalar = np.maximum(v_rad_scalar, 0.0)              # 只阻尼向外速度
            force -= self.k_damp * v_rad_scalar[:, None] * radial_unit

            # 累加到 xfrc_applied[body, :3]
            for i, bid in enumerate(body_ids_arr):
                if mask[i]:
                    data.xfrc_applied[bid, 0] += force[i, 0]
                    data.xfrc_applied[bid, 1] += force[i, 1]
                    data.xfrc_applied[bid, 2] += force[i, 2]

        # 中心吸引力(所有 body,无论是否越界)— 让导丝沿中心线走
        if self.k_attract > 0.0:
            attract = -self.k_attract * radial_vec   # 朝最近 centerline 点
            for i, bid in enumerate(body_ids_arr):
                data.xfrc_applied[bid, 0] += attract[i, 0]
                data.xfrc_applied[bid, 1] += attract[i, 1]
                data.xfrc_applied[bid, 2] += attract[i, 2]

        tip_idx = int(idxs[-1])
        tip_s = float(self.s[tip_idx])

        return {
            "max_radial_error": max_err,
            "num_projected_bodies": num_proj,
            "tip_centerline_s": tip_s,
        }
