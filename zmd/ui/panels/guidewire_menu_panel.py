"""导丝菜单 — 独立侧边栏页,包含参数 / 控制 / MuJoCo 物理参数。

只支持 MuJoCo 医疗导丝(其它类型已删除)。
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QColorDialog, QFrame, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from zmd.simulation.guidewire.mujoco_config import MujocoGuidewireConfig
from zmd.ui.panels.control_sidebar import (
    FocusWheelDoubleSpinBox,
    make_hold_button, make_section_label, make_step_button,
)


class GuidewireMenuPanel(QFrame):
    """导丝菜单页 — 单一 MuJoCo 导丝,所有控制都广播到 workspace 走同步规则。"""

    radius_changed = pyqtSignal(float)
    color_changed = pyqtSignal(str)             # hex color
    advance_clicked = pyqtSignal()              # 单步前进
    retract_clicked = pyqtSignal()              # 单步后退
    rotate_cw_clicked = pyqtSignal()
    rotate_ccw_clicked = pyqtSignal()
    hold_forward_pressed = pyqtSignal()
    hold_backward_pressed = pyqtSignal()
    hold_released = pyqtSignal()
    reset_requested = pyqtSignal()
    inset_toggle_changed = pyqtSignal(bool)
    minimap_toggle_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QFrame] = None):
        super().__init__(parent)
        self._suspend = False

        # ---- 1. 参数 ----
        self.radius_input = FocusWheelDoubleSpinBox()
        self.radius_input.setRange(0.01, 1.0); self.radius_input.setDecimals(3)
        self.radius_input.setSingleStep(0.01)
        self.radius_input.setValue(float(MujocoGuidewireConfig.radius))
        self.radius_input.valueChanged.connect(self._on_radius_changed)

        self.color_button = QPushButton("Choose Color")
        self.color_button.setObjectName("SidebarStepButton")
        self.color_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.color_button.clicked.connect(self._on_color_clicked)
        self.color_swatch = QLabel()
        self.color_swatch.setFixedSize(20, 20)
        self._current_color = "#7dd3fc"
        self._update_color_swatch()

        # ---- 2. 控制 ----
        self.advance_btn = make_step_button("Forward →", "Move forward one step")
        self.retract_btn = make_step_button("← Back", "Move back one step")
        self.hold_fwd_btn = make_hold_button("Forward ▶▶", "Hold to move forward continuously")
        self.hold_back_btn = make_hold_button("◀◀ Back", "Hold to move backward continuously")
        self.reset_btn = make_step_button("⟲ Reset", "Return the guidewire to the entry")

        self.advance_btn.clicked.connect(self.advance_clicked.emit)
        self.retract_btn.clicked.connect(self.retract_clicked.emit)
        self.hold_fwd_btn.pressed.connect(self.hold_forward_pressed.emit)
        self.hold_fwd_btn.released.connect(self.hold_released.emit)
        self.hold_back_btn.pressed.connect(self.hold_backward_pressed.emit)
        self.hold_back_btn.released.connect(self.hold_released.emit)
        self.reset_btn.clicked.connect(self.reset_requested.emit)

        # 旋转按钮(MuJoCo 支持 twist + J-tip steer)
        self.rotate_ccw_btn = make_step_button("↺ CCW", "Rotate the tip counterclockwise")
        self.rotate_cw_btn = make_step_button("CW ↻", "Rotate the tip clockwise")
        self.rotate_ccw_btn.clicked.connect(self.rotate_ccw_clicked.emit)
        self.rotate_cw_btn.clicked.connect(self.rotate_cw_clicked.emit)

        # ---- 3. 观察 ----
        self.inset_check = QCheckBox("Guidewire Inset")
        self.inset_check.setChecked(False)
        self.inset_check.toggled.connect(self.inset_toggle_changed.emit)
        self.minimap_check = QCheckBox("Bottom-Left Minimap")
        self.minimap_check.setChecked(True)
        self.minimap_check.toggled.connect(self.minimap_toggle_changed.emit)

        self._build_layout()

    def _build_layout(self):
        # 外层用滚动 — 避免参数太多挤压主窗口
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        scroll = QScrollArea(self)
        scroll.setObjectName("SidebarScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        outer.addWidget(scroll)
        inner = QWidget(); inner.setObjectName("SidebarScrollContent")
        scroll.setWidget(inner)
        l = QVBoxLayout(inner)
        l.setContentsMargins(0, 0, 6, 0); l.setSpacing(8)

        # ---- 导丝控制 section ----
        self.guidewire_section = QFrame()
        gl = QVBoxLayout(self.guidewire_section)
        gl.setContentsMargins(0, 0, 0, 0); gl.setSpacing(10)

        gl.addWidget(make_section_label("Guidewire Thickness (tube radius)"))
        gl.addWidget(self.radius_input)

        self.color_section_label = make_section_label("Guidewire Color")
        gl.addWidget(self.color_section_label)
        self.color_row_widget = QFrame()
        color_row = QHBoxLayout(self.color_row_widget)
        color_row.setContentsMargins(0, 0, 0, 0); color_row.setSpacing(8)
        color_row.addWidget(self.color_button)
        color_row.addWidget(self.color_swatch)
        color_row.addStretch(1)
        gl.addWidget(self.color_row_widget)

        gl.addWidget(make_section_label("Single-Step Controls"))
        row1 = QHBoxLayout(); row1.setContentsMargins(0, 0, 0, 0); row1.setSpacing(6)
        row1.addWidget(self.retract_btn, 1); row1.addWidget(self.advance_btn, 1)
        gl.addLayout(row1)

        gl.addWidget(make_section_label("Continuous Motion (hold)"))
        row2 = QHBoxLayout(); row2.setContentsMargins(0, 0, 0, 0); row2.setSpacing(6)
        row2.addWidget(self.hold_back_btn, 1); row2.addWidget(self.hold_fwd_btn, 1)
        gl.addLayout(row2)

        self.rotate_section_label = make_section_label("Rotation Controls")
        gl.addWidget(self.rotate_section_label)
        self.rotate_row_widget = QFrame()
        rot_row = QHBoxLayout(self.rotate_row_widget)
        rot_row.setContentsMargins(0, 0, 0, 0); rot_row.setSpacing(6)
        rot_row.addWidget(self.rotate_ccw_btn); rot_row.addWidget(self.rotate_cw_btn); rot_row.addStretch(1)
        gl.addWidget(self.rotate_row_widget)

        row3 = QHBoxLayout(); row3.setContentsMargins(0, 0, 0, 0); row3.setSpacing(6)
        row3.addWidget(self.reset_btn); row3.addStretch(1)
        gl.addLayout(row3)

        l.addWidget(self.guidewire_section)

        # 末尾 stretch 把多余高度顶到底部
        l.addStretch(1)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def set_radius(self, value: float):
        self._suspend = True
        self.radius_input.setValue(float(value))
        self._suspend = False

    def set_color(self, hex_color: str):
        self._suspend = True
        self._current_color = hex_color
        self._update_color_swatch()
        self._suspend = False

    def set_inset_visible(self, visible: bool):
        self._suspend = True
        self.inset_check.setChecked(bool(visible))
        self._suspend = False

    def set_minimap_visible(self, visible: bool):
        self._suspend = True
        self.minimap_check.setChecked(bool(visible))
        self._suspend = False

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _on_radius_changed(self, v):
        if self._suspend: return
        self.radius_changed.emit(float(v))

    def _on_color_clicked(self):
        from PyQt6.QtGui import QColor
        col = QColorDialog.getColor(
            QColor(self._current_color), self, "Choose Guidewire Color"
        )
        if col.isValid():
            self._current_color = col.name()
            self._update_color_swatch()
            self.color_changed.emit(self._current_color)

    def _update_color_swatch(self):
        self.color_swatch.setStyleSheet(
            f"background: {self._current_color}; "
            "border: 1px solid rgba(148, 163, 184, 0.7); border-radius: 4px;"
        )
