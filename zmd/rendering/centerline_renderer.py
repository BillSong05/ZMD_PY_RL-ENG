"""中心线渲染器 — 将 Centerline 渲染为参考管状线，支持运行时改 tube 粗细、可见性、分支显示。

与 GuidewireRenderer 的区别：
  - CenterlineRenderer 显示的是血管解剖中心线（静态参考线）
  - GuidewireRenderer 显示的是导丝仿真状态（动态推进线）
"""
import numpy as np
import pyvista as pv


class CenterlineRenderer:
    """渲染 Centerline 为彩色参考管状线。"""

    def __init__(self, view):
        self.view = view
        self.centerline = None
        self.radius = 0.10
        self.visible = True
        self.show_all_branches = True
        self.main_actor = None
        self.branch_actors = []

    # ------------------------------------------------------------------
    # 加载 / 重绘
    # ------------------------------------------------------------------

    def load(self, centerline) -> bool:
        self.centerline = centerline
        return self._rebuild()

    def set_radius(self, radius: float) -> None:
        radius = max(0.01, float(radius))
        if abs(radius - self.radius) < 1e-4:
            return
        self.radius = radius
        self._rebuild()

    def set_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible == self.visible:
            return
        self.visible = visible
        self._apply_visibility()

    def set_show_all_branches(self, show: bool) -> None:
        show = bool(show)
        if show == self.show_all_branches:
            return
        self.show_all_branches = show
        self._apply_visibility()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _rebuild(self) -> bool:
        self.clear()
        if self.centerline is None:
            return False
        pts = np.asarray(self.centerline.points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            return False
        try:
            # 浅色显示其它分支
            for branch in getattr(self.centerline, "branches", []) or []:
                bpts = np.asarray(branch.points, dtype=float)
                if len(bpts) < 2:
                    continue
                if len(bpts) == len(pts) and np.allclose(bpts, pts):
                    continue
                bpoly = pv.PolyData(bpts)
                bpoly.lines = np.hstack([[len(bpts)], np.arange(len(bpts))])
                btube = bpoly.tube(radius=self.radius * 0.7, n_sides=10)
                actor = self.view.plotter.add_mesh(
                    btube, color="#94a3b8", smooth_shading=True,
                    opacity=0.55, render=False,
                )
                self.branch_actors.append(actor)

            poly = pv.PolyData(pts)
            poly.lines = np.hstack([[len(pts)], np.arange(len(pts))])
            tube = poly.tube(radius=self.radius, n_sides=12)
            self.main_actor = self.view.plotter.add_mesh(
                tube, color="#f59e0b", specular=0.3, specular_power=16,
                smooth_shading=True, opacity=0.85, render=False,
            )
            self._apply_visibility()
            self.view.plotter.render()
            return True
        except Exception:
            return False

    def _apply_visibility(self):
        if self.main_actor is not None:
            self.main_actor.SetVisibility(1 if self.visible else 0)
        show_branches = self.visible and self.show_all_branches
        for actor in self.branch_actors:
            actor.SetVisibility(1 if show_branches else 0)
        self.view.plotter.render()

    def clear(self) -> None:
        if self.main_actor is not None:
            self.view.plotter.remove_actor(self.main_actor, render=False)
            self.main_actor = None
        for actor in self.branch_actors:
            self.view.plotter.remove_actor(actor, render=False)
        self.branch_actors = []
        self.view.plotter.render()
