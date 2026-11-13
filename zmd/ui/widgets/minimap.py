"""左下角透明 minimap — QPainter 2D 绘制中心线 + 当前导丝 + tip。

不用 VTK，纯 Qt 绘图，开销极低。把 (X, Y) 平面投影展示，附 Tip 坐标 / 进度 / 距终点。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class CenterlineMinimap(QWidget):
    """显示路径 + tip + 进度文本。"""

    DEFAULT_WIDTH = 280
    DEFAULT_HEIGHT = 220
    # 不再 setFixedSize；由 workspace _reposition_all_overlays 动态调整

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # 不用 WA_TranslucentBackground —— QtInteractor 上层 native 渲染时透明属性可能失效；
        # 改成在 paintEvent 里画半透明矩形作为背景，效果一样且兼容好
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        self._path: Optional[np.ndarray] = None
        self._inserted_idx: int = 0
        self._distance_to_end: float = 0.0
        # ball 模式：若不为 None 则用它当 tip（覆盖 path[inserted_idx]），
        # 同时改文字标签 "导丝头部" → "调试小球"
        self._actor_pos_override: Optional[np.ndarray] = None
        self._actor_label: str = "Guidewire Tip"

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def set_path(self, path_points: Optional[np.ndarray]) -> None:
        if path_points is None or len(path_points) < 2:
            self._path = None
        else:
            self._path = np.asarray(path_points, dtype=float)
        self.update()

    def set_progress(self, inserted_idx: int, distance_to_end: float) -> None:
        self._inserted_idx = int(inserted_idx)
        self._distance_to_end = float(distance_to_end)
        self.update()

    def set_actor_override(self, world_pos: Optional[np.ndarray],
                        label: str = "Debug Ball") -> None:
        """Ball 模式专用 — 显示实际世界位置作为 tip 标记 + 改文字标签。

        传 None 关闭，恢复导丝模式。
        """
        if world_pos is None:
            self._actor_pos_override = None
            self._actor_label = "Guidewire Tip"
        else:
            self._actor_pos_override = np.asarray(world_pos, dtype=float).reshape(3).copy()
            self._actor_label = str(label)
        self.update()

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()

        # 不透明度提高，避免穿过到 home page
        bg = QColor(7, 13, 24, 235)
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(QColor(96, 165, 250, 200), 1))
        painter.drawRoundedRect(QRectF(0, 0, w - 1, h - 1), 10, 10)

        text_area = 70
        plot_w, plot_h = w - 24, h - text_area - 24
        if self._path is None or len(self._path) < 2:
            painter.setPen(QColor(180, 200, 230, 200))
            painter.drawText(QRectF(12, 12, w - 24, h - 24),
                             int(Qt.AlignmentFlag.AlignCenter),
                     "Guidewire path not loaded")
            painter.end()
            return

        pts = self._path
        x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
        dx = max(x_max - x_min, 1e-3)
        dy = max(y_max - y_min, 1e-3)
        scale = min(plot_w / dx, plot_h / dy) * 0.92
        offset_x = 12 + (plot_w - dx * scale) / 2.0
        offset_y = 12 + (plot_h - dy * scale) / 2.0

        def to_screen(p):
            sx = offset_x + (p[0] - x_min) * scale
            # Y 翻转，上方为 +Y
            sy = offset_y + plot_h - (p[1] - y_min) * scale
            return QPointF(sx, sy)

        # 路径整体 — 浅灰
        painter.setPen(QPen(QColor(148, 163, 184, 200), 1.5))
        for i in range(1, len(pts)):
            painter.drawLine(to_screen(pts[i - 1]), to_screen(pts[i]))

        # 已推进导丝 — 青蓝色
        n_done = max(0, min(self._inserted_idx + 1, len(pts)))
        if n_done >= 2:
            painter.setPen(QPen(QColor(56, 189, 248, 235), 2.4))
            for i in range(1, n_done):
                painter.drawLine(to_screen(pts[i - 1]), to_screen(pts[i]))

        # Tip — 优先用 ball override 的真实世界位置，没有则用 path[inserted_idx]
        if self._actor_pos_override is not None:
            tip_world = self._actor_pos_override
        elif 0 <= self._inserted_idx < len(pts):
            tip_world = pts[self._inserted_idx]
        else:
            tip_world = None

        if tip_world is not None:
            tip_pos = to_screen(tip_world)
            # ball 模式用 cyan，导丝模式用 yellow
            is_ball = self._actor_pos_override is not None
            fill = QColor(34, 211, 238, 245) if is_ball else QColor(250, 204, 21, 245)
            ring = QColor(34, 211, 238, 130) if is_ball else QColor(250, 204, 21, 130)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(QColor(255, 255, 255, 220), 1))
            r = 4.5
            painter.drawEllipse(tip_pos, r, r)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(ring, 1))
            painter.drawEllipse(tip_pos, r * 2.2, r * 2.2)

            painter.setPen(QColor(220, 231, 246, 240))
            painter.setFont(QFont("Microsoft YaHei UI", 9, QFont.Weight.DemiBold))
            text_y = h - text_area + 8
            line_height = 18
            painter.drawText(14, text_y + line_height * 1,
                f"{self._actor_label} ({tip_world[0]:+.1f}, {tip_world[1]:+.1f}, {tip_world[2]:+.1f})")
            total = max(len(pts) - 1, 1)
            progress = (self._inserted_idx / total) * 100.0
            painter.drawText(14, text_y + line_height * 2,
                         f"Progress {progress:5.1f}%")
            painter.drawText(14, text_y + line_height * 3,
                         f"Distance to End {self._distance_to_end:5.1f} mm")

        painter.end()
