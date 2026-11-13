"""血管菜单 — 右侧弹出面板，承载血管相关属性 / 物理开关。

当前内容：
* 静态墙壁碰撞 (STL 不可穿透) — 调试小球的 SDF 拒绝开关；未来真正物理后端
  也接入这里
* 占位：血管透明度 / 是否显示血管壁 / 切面剖面等（后续接）

与左侧 ControlSidebar 等价的 toggle 模型 —— workspace 顶部栏按钮控制展开/折叠。
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget,
)

from PyQt6.QtWidgets import QHBoxLayout

from zmd.ui.panels.control_sidebar import (
    FocusWheelDoubleSpinBox, SIDEBAR_STYLE, make_section_label,
)


class VesselMenuPanel(QFrame):
    """右侧弹出血管菜单 —— 视觉风格沿用左侧 ControlSidebar 的 #WorkbenchSidebar。"""

    # 调试小球 / 医疗导丝静态墙壁碰撞总开关
    static_collision_changed = pyqtSignal(bool)
    # 血管透明度
    vessel_opacity_changed = pyqtSignal(float)

    DEFAULT_WIDTH = 280

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # 复用 WorkbenchSidebar objectName 让 SIDEBAR_STYLE 自动套上同一套
        # 圆角 / 蓝色边框 / 半透明背景 / scroll bar / checkbox 等子控件样式
        self.setObjectName("WorkbenchSidebar")
        self.setStyleSheet(SIDEBAR_STYLE)
        self.expanded = False
        self.expanded_width = self.DEFAULT_WIDTH
        self._build_ui()
        self.setFixedWidth(0)
        self.setVisible(False)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 12, 12); outer.setSpacing(10)

        # 标题用 SidebarTitle objectName，自动套上左 sidebar 同款字体
        title = QLabel("Vessel")
        title.setObjectName("SidebarTitle")
        outer.addWidget(title)

        scroll = QScrollArea(self)
        scroll.setObjectName("SidebarScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        outer.addWidget(scroll, 1)

        inner = QWidget()
        inner.setObjectName("SidebarScrollContent")
        scroll.setWidget(inner)
        l = QVBoxLayout(inner)
        l.setContentsMargins(0, 0, 6, 0); l.setSpacing(8)

        # ---- 物理 ----
        l.addWidget(make_section_label("Basic Physics"))
        self.collision_check = QCheckBox("Static wall collision (solid STL)")
        self.collision_check.setChecked(True)
        self.collision_check.toggled.connect(self.static_collision_changed.emit)
        l.addWidget(self.collision_check)

        # 透明度（即时生效，会影响渲染）
        self.opacity_input = FocusWheelDoubleSpinBox()
        self.opacity_input.setRange(0.0, 1.0); self.opacity_input.setDecimals(2)
        self.opacity_input.setSingleStep(0.05); self.opacity_input.setValue(0.45)
        self.opacity_input.valueChanged.connect(self.vessel_opacity_changed.emit)
        l.addLayout(self._row("Vessel Opacity", self.opacity_input))

        # 碰撞管线 / 摩擦 / contact distance 等 UI 已移除 — 物理后端未接入

        l.addStretch(1)

    def _row(self, label_text, widget):
        row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(8)
        lab = QLabel(label_text); lab.setStyleSheet("color: #94a3b8; font-size: 11px;")
        lab.setMinimumWidth(110); lab.setMaximumWidth(120)
        row.addWidget(lab); row.addWidget(widget, 1)
        return row

    # ------------------------------------------------------------------
    # toggle API
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        self.expanded = not self.expanded
        self.apply_state()

    def apply_state(self) -> None:
        self.setVisible(self.expanded)
        self.setFixedWidth(self.expanded_width if self.expanded else 0)

    def set_static_collision(self, enabled: bool) -> None:
        self.collision_check.blockSignals(True)
        self.collision_check.setChecked(bool(enabled))
        self.collision_check.blockSignals(False)
