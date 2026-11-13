"""导丝渲染器 — 根据中心线生成管状 mesh 并添加到 PyVista 视窗。

tube/line 模式仍可单独切换（line 模式给 inset 跟随相机用，已不强依赖）。
"""
import numpy as np
import pyvista as pv


GUIDEWIRE_STYLES = {
    "centerline": {
        "color": "#7dd3fc",          # 调试青蓝
        "default_radius": 0.03,
        "specular": 0.35,
        "specular_power": 18,
        "ambient": 0.25,
        "diffuse": 0.75,
        "smooth_shading": True,
        "n_sides": 14,
        "opacity": 1.0,
        "color_overridable": True,
    },
}


class GuidewireRenderer:
    """导丝渲染器，将中心线坐标转为 PyVista tube 或 line 并渲染。"""

    def __init__(self, view, style_name: str = "centerline"):
        self.view = view
        self.actor = None
        self.tip_marker_actor = None       # 默认不再创建大球 marker
        self._last_centerline = None
        self.visible = True
        self.line_mode = False
        self.show_tip_marker = False       # 关闭大球；医疗导丝完全不画 tip marker
        self._apply_style(style_name, rebuild=False)

    # ------------------------------------------------------------------
    # 样式
    # ------------------------------------------------------------------

    def _apply_style(self, style_name: str, rebuild: bool = True) -> None:
        style_name = style_name if style_name in GUIDEWIRE_STYLES else "centerline"
        self.style_name = style_name
        self.style = GUIDEWIRE_STYLES[style_name]
        # 切 style 时强制回到 style 默认配色/半径，避免上一类型残留
        self.color = self.style["color"]
        self.radius = float(self.style["default_radius"])
        if rebuild:
            # 切 style 必须重建 actor，clear 旧的 specular / tip marker
            self.clear()
            if self._last_centerline is not None:
                self._rebuild(self._last_centerline)

    def set_style(self, style_name: str) -> None:
        if style_name == getattr(self, "style_name", None):
            return
        self._apply_style(style_name, rebuild=True)

    # ------------------------------------------------------------------
    # 参数
    # ------------------------------------------------------------------

    def set_radius(self, radius: float) -> None:
        radius = max(0.01, float(radius))
        if abs(radius - self.radius) < 1e-4:
            return
        self.radius = radius
        if self._last_centerline is not None and not self.line_mode:
            self.update_centerline(self._last_centerline, radius=radius)

    def set_color(self, color: str) -> None:
        """更新导丝主体颜色 — 医疗 style 锁色，忽略此调用。"""
        if not self.style.get("color_overridable", True):
            return
        color = str(color)
        if color == self.color:
            return
        self.color = color
        if self.actor is not None:
            try:
                from PyQt6.QtGui import QColor
                c = QColor(color)
                if c.isValid():
                    self.actor.GetProperty().SetColor(c.redF(), c.greenF(), c.blueF())
                    self.view.plotter.render()
            except Exception:
                pass

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        if self.actor is not None:
            self.actor.SetVisibility(1 if self.visible else 0)
        if self.tip_marker_actor is not None:
            self.tip_marker_actor.SetVisibility(1 if self.visible else 0)
        self.view.plotter.render()

    def set_line_mode(self, enabled: bool) -> None:
        """切换 tube / line 渲染。会立即用上次的 centerline 重绘。"""
        enabled = bool(enabled)
        if enabled == self.line_mode:
            return
        self.line_mode = enabled
        if self._last_centerline is not None:
            self._rebuild(self._last_centerline)

    # ------------------------------------------------------------------
    # 加载 / 更新
    # ------------------------------------------------------------------

    def load_from_centerline(self, centerline: np.ndarray, radius: float = 0.15) -> None:
        self.update_centerline(centerline, radius=radius)

    def update_centerline(self, centerline: np.ndarray, radius: float = 0.15) -> None:
        centerline = np.asarray(centerline, dtype=float)
        if centerline.ndim != 2 or centerline.shape[1] != 3:
            self.clear()
            return
        self._last_centerline = centerline.copy()
        self.radius = float(radius)
        if len(centerline) == 0:
            self.clear()
            return
        if len(centerline) == 1:
            self._rebuild_tip_only(centerline[0])
            return
        self._rebuild(centerline)

    def _rebuild_tip_only(self, point: np.ndarray) -> None:
        if self.actor is not None:
            self.view.plotter.remove_actor(self.actor, render=False)
            self.actor = None
        if self.tip_marker_actor is not None:
            self.view.plotter.remove_actor(self.tip_marker_actor, render=False)
            self.tip_marker_actor = None
        try:
            point = np.asarray(point, dtype=float).reshape(3)
            marker_radius = max(self.radius * 1.2, 0.02)
            self.actor = self.view.plotter.add_mesh(
                pv.Sphere(
                    radius=marker_radius,
                    center=tuple(point),
                    theta_resolution=8,
                    phi_resolution=8,
                ),
                color=self.color,
                specular=float(self.style["specular"]),
                specular_power=float(self.style["specular_power"]),
                ambient=float(self.style["ambient"]),
                diffuse=float(self.style["diffuse"]),
                opacity=float(self.style.get("opacity", 1.0)),
                smooth_shading=bool(self.style["smooth_shading"]),
                render=False,
            )
        except Exception:
            self.actor = self.view.plotter.add_mesh(
                pv.PolyData(np.asarray([point], dtype=float)),
                color=self.color,
                point_size=6,
                render_points_as_spheres=True,
                render=False,
            )
        if self.actor is not None:
            self.actor.SetVisibility(1 if self.visible else 0)
        self.view.plotter.render()

    # ------------------------------------------------------------------
    # 渲染平滑（仅可视化，不改物理点 / tip / 进度）
    # ------------------------------------------------------------------

    def _smooth_polyline(self, pts: np.ndarray):
        """把离散导丝点沿原始折线线性加密成 PolyData(不外扩、不过冲)。

        关键改动(修穿模视觉问题):不再用 pv.Spline 的三次插值,
        因为它在弯道处会 overshoot(渲染曲线鼓出血管壁,看起来穿模)。
        改用沿原始折线的线性插值加密:新点严格落在原折线段上,
        永远不会比物理折线更外扩,既减少节点感,又不会穿壁。

        - 保留入口点(首)和 tip(末),端点精确通过,不偏移。
        - 去掉重复/过近点,避免数值抖动。
        - 点太少(<2)时返回 None,调用方回退到原始离散渲染。
        - 只影响显示,不参与物理/碰撞/进度。
        """
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            return None

        # 去过近点,但首点和末点(tip)必须保留
        min_d = max(self.radius * 0.5, 1e-4)
        cleaned = [pts[0]]
        for p in pts[1:-1]:
            if np.linalg.norm(p - cleaned[-1]) >= min_d:
                cleaned.append(p)
        tip = pts[-1]
        if len(cleaned) > 1 and np.linalg.norm(tip - cleaned[-1]) < min_d:
            cleaned[-1] = tip          # 末点过近 → 用 tip 替换前点,保证 tip 精确
        else:
            cleaned.append(tip)
        cleaned = np.asarray(cleaned, dtype=float)
        if len(cleaned) < 2:
            return None                # 点太少,交给原始渲染

        # 角点圆滑:Chaikin 切角(只往内收,不外扩,不会穿壁)。
        # 修"渲染不平滑":线性加密保留了原始折点处的尖角,弯道看起来一节一节;
        # Chaikin 反复用 1/4、3/4 点替换每段,把折点磨成圆角,而新点都是相邻原
        # 点的凸组合 → 始终落在原折线内侧,永不比物理折线更外扩(不像三次样条会过冲)。
        smoothed = self._chaikin(cleaned, iterations=2)

        # 沿折线按 ~0.3mm 弧长线性加密(不过冲版)
        dense = self._densify_polyline(smoothed, spacing=0.3, max_points=1200)
        if dense is None or len(dense) < 2:
            return None
        poly = pv.PolyData(dense)
        poly.lines = np.hstack([[len(dense)], np.arange(len(dense))])
        return poly

    @staticmethod
    def _chaikin(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
        """Chaikin 切角圆滑(开放折线版):首点(入口)、末点(tip)固定不动。

        每次迭代把每段 [P_i, P_{i+1}] 替换为 Q=0.75 P_i+0.25 P_{i+1} 与
        R=0.25 P_i+0.75 P_{i+1} 两点,折点被磨圆。新点均为相邻原点凸组合,
        只会落在原折线内侧,不外扩 → 不会让渲染管穿出血管壁。
        """
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 3:
            return pts
        for _ in range(max(0, int(iterations))):
            if len(pts) < 3:
                break
            q = 0.75 * pts[:-1] + 0.25 * pts[1:]
            r = 0.25 * pts[:-1] + 0.75 * pts[1:]
            inner = np.empty((2 * (len(pts) - 1), 3), dtype=float)
            inner[0::2] = q
            inner[1::2] = r
            pts = np.vstack([pts[0], inner, pts[-1]])
        return pts

    @staticmethod
    def _densify_polyline(pts: np.ndarray, spacing: float = 0.3,
                          max_points: int = 1200) -> np.ndarray:
        """沿折线按固定弧长间隔线性插值加密,端点精确保留,严格落在原折线上。"""
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(seg.sum())
        if total < 1e-9:
            return pts
        spacing = max(1e-3, float(spacing))
        n = int(min(max_points, max(len(pts), round(total / spacing) + 1)))
        # 在 [0, total] 上均匀取 n 个弧长采样,逐段线性插值
        s_targets = np.linspace(0.0, total, n)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        out = np.zeros((n, 3), dtype=float)
        j = 0
        for i, s in enumerate(s_targets):
            while j < len(seg) - 1 and cum[j + 1] < s:
                j += 1
            seg_len = max(cum[j + 1] - cum[j], 1e-9)
            t = (s - cum[j]) / seg_len
            t = min(max(t, 0.0), 1.0)
            out[i] = pts[j] + t * (pts[j + 1] - pts[j])
        out[0] = pts[0]
        out[-1] = pts[-1]
        return out

    def _rebuild(self, centerline: np.ndarray) -> None:
        if self.actor is not None:
            self.view.plotter.remove_actor(self.actor, render=False)
            self.actor = None
        if self.tip_marker_actor is not None:
            self.view.plotter.remove_actor(self.tip_marker_actor, render=False)
            self.tip_marker_actor = None
        s = self.style
        try:
            # 渲染前先平滑(失败/点太少则回退到离散折线)
            smooth = self._smooth_polyline(centerline)
            if smooth is not None:
                render_poly = smooth
            else:
                render_poly = pv.PolyData(centerline)
                render_poly.lines = np.hstack(
                    [[len(centerline)], np.arange(len(centerline))]
                )
            if self.line_mode:
                self.actor = self.view.plotter.add_mesh(
                    render_poly, color=self.color, line_width=4,
                    render_lines_as_tubes=False, render=False,
                )
            else:
                tube = render_poly.tube(radius=self.radius, n_sides=int(s["n_sides"]))
                self.actor = self.view.plotter.add_mesh(
                    tube, color=self.color,
                    specular=float(s["specular"]),
                    specular_power=float(s["specular_power"]),
                    ambient=float(s["ambient"]),
                    diffuse=float(s["diffuse"]),
                    opacity=float(s.get("opacity", 1.0)),
                    smooth_shading=bool(s["smooth_shading"]),
                    render=False,
                )
            # 默认两套 style 都关闭 tip 大球 marker
            if self.show_tip_marker:
                tip = centerline[-1]
                marker_radius = self.radius * 1.25
                self.tip_marker_actor = self.view.plotter.add_mesh(
                    pv.Sphere(radius=marker_radius, center=tuple(tip),
                              theta_resolution=10, phi_resolution=10),
                    color="#facc15", specular=0.4, render=False,
                )
        except Exception:
            self.actor = self.view.plotter.add_mesh(
                pv.PolyData(centerline), color=self.color,
                line_width=2, render=False,
            )
        if self.actor is not None:
            self.actor.SetVisibility(1 if self.visible else 0)
        if self.tip_marker_actor is not None:
            self.tip_marker_actor.SetVisibility(1 if self.visible else 0)
        self.view.plotter.render()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def clear(self) -> None:
        if self.actor is not None:
            self.view.plotter.remove_actor(self.actor, render=False)
            self.actor = None
        if self.tip_marker_actor is not None:
            self.view.plotter.remove_actor(self.tip_marker_actor, render=False)
            self.tip_marker_actor = None
        self._last_centerline = None
        self.view.plotter.render()
