"""右上角导丝观察小窗 — QtInteractor 子窗口，专门近距离观察当前选中导丝头部运动。

主场景相机不动；点击 follow 后这个小窗显示并跟随导丝 tip + tangent。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pyvista as pv
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor


INSET_STYLES = {
    "centerline": {
        "color": "#7dd3fc",
        "min_radius": 0.04,
        "specular": 0.35,
        "specular_power": 18,
        "smooth_shading": True,
        "n_sides": 10,
        "color_overridable": True,
    },
    "medical": {
        "color": "#b8bcc7",
        "min_radius": 0.05,
        "specular": 0.95,
        "specular_power": 70,
        "smooth_shading": True,
        "n_sides": 20,
        "color_overridable": False,
    },
}


class GuidewireInsetView(QFrame):
    """200×150 小型 3D 观察窗 — 含血管 + 导丝 line + tip marker，自动随 tip 旋转。"""

    # 视距按 polyline 实际跨度自适应，下面是上下限
    MIN_BACK_DISTANCE = 2.0
    MAX_BACK_DISTANCE = 12.0
    MIN_LOOKAHEAD = 4.0
    MAX_LOOKAHEAD = 20.0

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("GuidewireInsetView")
        self.setStyleSheet(
            "#GuidewireInsetView { "
            "background: rgba(7, 13, 24, 0.96); "
            "border: 1px solid rgba(96, 165, 250, 0.75); "
            "border-radius: 10px; "
            "}"
        )
        # 默认尺寸 — 由 workspace _reposition_all_overlays 动态 setFixedSize
        self.resize(280, 200)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.plotter = QtInteractor(self)
        self._shutting_down = False
        self.plotter.interactor.setSizePolicy(QSizePolicy.Policy.Expanding,
                                              QSizePolicy.Policy.Expanding)
        layout.addWidget(self.plotter.interactor, 1)
        self.plotter.set_background("#0d1116", top="#1a1f2e")
        try:
            head_light = pv.Light(light_type="cameralight", intensity=0.45)
            self.plotter.add_light(head_light)
        except Exception:
            pass
        # 禁用交互，避免用户点击小窗影响主操作
        try:
            self.plotter.iren.GetInteractorStyle().SetMotionFactor(0.0)
            self.plotter.iren.Disable()
        except Exception:
            pass

        self.vessel_actor = None
        self.guidewire_actor = None
        self.tip_marker_actor = None
        self.style_name = "centerline"
        self.style = INSET_STYLES[self.style_name]
        self.guidewire_color = self.style["color"]
        self.tip_color = "#facc15"

        # 未插入占位文本（覆盖在小窗上）
        self.placeholder_label = QLabel("Guidewire not inserted", self)
        self.placeholder_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.placeholder_label.setStyleSheet(
            "color: #94a3b8; background: rgba(7, 13, 24, 0.85); "
            "border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 700;"
        )
        self.placeholder_label.adjustSize()
        self.placeholder_label.move(
            (self.width() - self.placeholder_label.width()) // 2,
            (self.height() - self.placeholder_label.height()) // 2,
        )
        self.placeholder_label.show()
        self.placeholder_label.raise_()

    def set_color(self, color: str) -> None:
        # 医疗 style 锁色，UI 颜色调整对小窗也不生效
        if not self.style.get("color_overridable", True):
            return
        self.guidewire_color = str(color)
        if self.guidewire_actor is not None:
            try:
                from PyQt6.QtGui import QColor
                c = QColor(color)
                if c.isValid():
                    self.guidewire_actor.GetProperty().SetColor(c.redF(), c.greenF(), c.blueF())
                    self.plotter.render()
            except Exception:
                pass

    def set_style(self, style_name: str) -> None:
        """切换 inset 渲染样式。会清掉旧 actor，下一帧用新 style 重绘。"""
        style_name = style_name if style_name in INSET_STYLES else "centerline"
        if style_name == self.style_name:
            return
        self.style_name = style_name
        self.style = INSET_STYLES[style_name]
        self.guidewire_color = self.style["color"]
        self.clear_guidewire()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def load_vessel(self, mesh: pv.PolyData) -> None:
        # 第一人称小窗:血管 100% 不透明(实心壁),内窥镜视角看实管腔。
        # backface culling 关掉,这样相机在管腔内时仍能看到内表面。
        if self.vessel_actor is not None:
            self.plotter.remove_actor(self.vessel_actor, render=False)
        try:
            self.vessel_actor = self.plotter.add_mesh(
                mesh, color="#bf4038", opacity=1.0,
                specular=0.12, ambient=0.30, diffuse=0.60,
                smooth_shading=True, render=False, reset_camera=False,
            )
            try:
                self.vessel_actor.GetProperty().SetBackfaceCulling(0)
            except Exception:
                pass
            # 与主视角同样的 0.1 缩放
            self.vessel_actor.SetScale(0.1, 0.1, 0.1)
        except Exception:
            self.vessel_actor = None
        # 不要 plotter.render() 触发 auto-reset,等 update_guidewire 设好相机再 render

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 居中放占位文本
        if hasattr(self, "placeholder_label") and self.placeholder_label is not None:
            self.placeholder_label.move(
                (self.width() - self.placeholder_label.width()) // 2,
                (self.height() - self.placeholder_label.height()) // 2,
            )

    def update_guidewire(self, visible_points: np.ndarray, gw_radius: float = 0.05,
                         camera_pose: Optional[dict] = None) -> None:
        """重建小窗的导丝 line 并把相机移到 tip 后方。空 polyline 显示占位文本。

        camera_pose: 可选 {position, focal_point, up}(世界坐标),由后端计算并保证
        相机在血管腔内、沿中心线方向看。提供时优先使用,否则退回 tip 切向自算。
        """
        if visible_points is None or len(visible_points) < 2:
            self.clear_guidewire()
            return
        # 隐藏占位
        self.placeholder_label.hide()
        pts = np.asarray(visible_points, dtype=float)
        if not np.all(np.isfinite(pts)):
            return

        # 清旧 actor
        if self.guidewire_actor is not None:
            self.plotter.remove_actor(self.guidewire_actor, render=False)
            self.guidewire_actor = None
        if self.tip_marker_actor is not None:
            self.plotter.remove_actor(self.tip_marker_actor, render=False)
            self.tip_marker_actor = None

        # 整根 polyline + 较粗 line + tube 双绘 —— 短鸿丝肉眼可见
        n = len(pts)
        poly = pv.PolyData(pts)
        poly.lines = np.hstack([[n], np.arange(n)])
        s = self.style
        try:
            # tube：按 style 配色和材质参数；medical = 银灰金属、高 specular
            tube_radius = max(float(gw_radius), float(s["min_radius"]))
            tube = poly.tube(radius=tube_radius, n_sides=int(s["n_sides"]))
            self.guidewire_actor = self.plotter.add_mesh(
                tube, color=self.guidewire_color,
                specular=float(s["specular"]),
                specular_power=float(s["specular_power"]),
                smooth_shading=bool(s["smooth_shading"]),
                render=False, reset_camera=False,
            )
        except Exception:
            self.guidewire_actor = self.plotter.add_mesh(
                poly, color=self.guidewire_color, line_width=6,
                render_lines_as_tubes=False, render=False, reset_camera=False,
            )

        # ---- 第一人称相机 ----
        # 优先使用后端提供的腔内安全相机位姿(SDF safe + 沿中心线方向);
        # 没提供时退回 tip 切向自算(旧逻辑)。
        cam_pos = focal = up = None
        if camera_pose is not None:
            try:
                cam_pos = np.asarray(camera_pose["position"], dtype=float)
                focal = np.asarray(camera_pose["focal_point"], dtype=float)
                up = np.asarray(camera_pose["up"], dtype=float)
                if not (np.all(np.isfinite(cam_pos)) and np.all(np.isfinite(focal))
                        and np.all(np.isfinite(up))):
                    cam_pos = focal = up = None
            except Exception:
                cam_pos = focal = up = None

        if cam_pos is None:
            # 退回:tip 与往回 4 个点的位移做切线
            tlb = min(4, n - 1)
            tangent = pts[-1] - pts[-1 - tlb]
            t_norm = float(np.linalg.norm(tangent))
            if t_norm < 1e-6:
                tangent = pts[-1] - pts[0]
                t_norm = float(np.linalg.norm(tangent))
            if t_norm < 1e-6:
                self.plotter.render()
                return
            t_hat = tangent / t_norm
            BACK_MM = 3.0
            LOOKAHEAD_MM = 10.0
            tip = pts[-1]
            cam_pos = tip - t_hat * BACK_MM
            focal = tip + t_hat * LOOKAHEAD_MM
            world_up = np.array([0.0, 0.0, 1.0])
            dot = float(np.dot(t_hat, world_up))
            up = world_up - dot * t_hat
            un = float(np.linalg.norm(up))
            if un < 1e-6:
                up = np.array([1.0, 0.0, 0.0])
                up = up - float(np.dot(up, t_hat)) * t_hat
                un = float(np.linalg.norm(up))
            up = up / un if un >= 1e-6 else np.array([0.0, 0.0, 1.0])

        try:
            cam = self.plotter.camera
            cam.position = tuple(np.asarray(cam_pos).tolist())
            cam.focal_point = tuple(np.asarray(focal).tolist())
            cam.up = tuple(np.asarray(up).tolist())
            # 较窄视角让 guidewire 占满更多像素
            cam.view_angle = 30.0
            # 局部放大跟随:固定 clipping(不随 SDF 动态改),近面不要太近,
            # 让导丝头 + 头附近血管 + 前方走向都能看清。
            try:
                cam.clipping_range = (0.1, 100.0)
            except Exception:
                try:
                    self.plotter.renderer.ResetCameraClippingRange()
                except Exception:
                    pass
        except Exception:
            pass
        self.plotter.render()

    def clear_guidewire(self) -> None:
        for a in (self.guidewire_actor, self.tip_marker_actor):
            if a is not None:
                try:
                    self.plotter.remove_actor(a, render=False)
                except Exception:
                    pass
        self.guidewire_actor = None
        self.tip_marker_actor = None
        # 显示"未插入导丝"占位
        if hasattr(self, "placeholder_label") and self.placeholder_label is not None:
            self.placeholder_label.show()
            self.placeholder_label.raise_()
            self.placeholder_label.adjustSize()
            self.placeholder_label.move(
                (self.width() - self.placeholder_label.width()) // 2,
                (self.height() - self.placeholder_label.height()) // 2,
            )
        self.plotter.render()

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            self.plotter.iren.Disable()
        except Exception:
            pass
        try:
            self.plotter.clear()
        except Exception:
            pass
        try:
            self.plotter.close()
        except Exception:
            pass
        try:
            self.plotter.interactor.close()
        except Exception:
            pass

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
