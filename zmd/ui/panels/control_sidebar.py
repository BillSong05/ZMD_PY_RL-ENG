# 职责：默认工作台实验场景控制侧边栏 — 按对象切换控制面板。
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QToolButton,
    QVBoxLayout,
)


SIDEBAR_STYLE = """
#WorkbenchSidebar {
    background: rgba(10, 35, 80, 0.42);
    border: 1px solid rgba(59, 130, 246, 0.48);
    border-radius: 16px;
}
#SidebarTitle { color: #dbeafe; font-size: 15px; font-weight: 900; }
#SidebarActiveScene { color: #86efac; font-size: 12px; font-weight: 800; padding: 0 6px; }
#SidebarHint, #SidebarSectionLabel { color: #93a4bb; font-size: 12px; font-weight: 700; }
#SidebarFieldLabel { color: #dbe7f6; font-size: 12px; font-weight: 700; }
#SidebarStepButton {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(96, 165, 250, 0.78);
    border-radius: 10px; color: #bfdbfe;
    font-size: 12px; font-weight: 900; padding: 4px 10px; min-width: 28px;
}
#SidebarStepButton:hover {
    background: rgba(37, 99, 235, 0.65);
    border-color: rgba(147, 197, 253, 0.95); color: #ffffff;
}
#SidebarStepButton:pressed { background: rgba(37, 99, 235, 0.92); }
#SidebarSyncButton {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(96, 165, 250, 0.78);
    border-radius: 12px; color: #bfdbfe;
    font-size: 12px; font-weight: 900; padding: 5px 12px;
}
#SidebarSyncButton:hover { background: rgba(37, 99, 235, 0.56); border-color: rgba(147, 197, 253, 0.95); color: #ffffff; }
#SidebarSyncButton:checked { background: rgba(37, 99, 235, 0.92); border-color: rgba(186, 230, 253, 1.0); color: #ffffff; }
#SidebarScrollArea { background: transparent; border: none; }
#SidebarScrollContent { background: transparent; }
QScrollBar:vertical { background: rgba(15, 23, 42, 0.32); border: none; border-radius: 5px; margin: 0; width: 10px; }
QScrollBar::handle:vertical { background: rgba(96, 165, 250, 0.78); border-radius: 5px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: rgba(147, 197, 253, 0.96); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; border: none; height: 0; }
QComboBox, QDoubleSpinBox {
    background: rgba(15, 23, 42, 0.72); border: 1px solid rgba(148, 163, 184, 0.38);
    border-radius: 8px; color: #eef6ff; min-height: 30px; padding: 3px 8px;
}
QComboBox:hover, QDoubleSpinBox:hover { border-color: rgba(147, 197, 253, 0.9); }
QCheckBox { color: #dbe7f6; font-size: 12px; font-weight: 700; spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; }
QCheckBox::indicator:unchecked { border: 1px solid rgba(148, 163, 184, 0.6); border-radius: 3px; background: rgba(15, 23, 42, 0.72); }
QCheckBox::indicator:checked { border: 1px solid rgba(96, 165, 250, 0.9); border-radius: 3px; background: rgba(37, 99, 235, 0.85); }
"""


class FocusWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self, event):
        if self.hasFocus(): super().wheelEvent(event)
        else: event.ignore()


# ----------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------

def make_section_label(text):
    lb = QLabel(text)
    lb.setObjectName("SidebarSectionLabel")
    lb.setWordWrap(True)
    lb.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    return lb


def make_step_button(text, tooltip):
    """单击按钮 — 点一下走一步，不自动重复。"""
    b = QToolButton()
    b.setObjectName("SidebarStepButton")
    b.setText(text); b.setToolTip(tooltip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


class HoldToStepEmitter(QObject):
    """长按按钮 — 按下后每 interval ms 发射一次 emit_fn()，松开停止。"""

    def __init__(self, button: QToolButton, on_tick, interval_ms: int = 90):
        super().__init__(button)
        self._on_tick = on_tick
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._on_tick)
        button.pressed.connect(self._start)
        button.released.connect(self._stop)

    def _start(self):
        self._on_tick()          # 按下瞬间先触发一次
        self._timer.start()

    def _stop(self):
        self._timer.stop()


def make_hold_button(text, tooltip):
    """长按按钮 — 按住持续触发，由 HoldToStepEmitter 接管。"""
    b = QToolButton()
    b.setObjectName("SidebarStepButton")
    b.setText(text); b.setToolTip(tooltip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def make_vector_inputs(mn, mx, step, on_change):
    inputs = []
    for _ in range(3):
        s = FocusWheelDoubleSpinBox()
        s.setRange(mn, mx); s.setDecimals(3); s.setSingleStep(step)
        s.valueChanged.connect(on_change)
        s.setMinimumWidth(190)
        inputs.append(s)
    return inputs


def make_vector_layout(inputs):
    l = QGridLayout()
    l.setContentsMargins(0, 0, 0, 0); l.setHorizontalSpacing(10); l.setVerticalSpacing(8); l.setColumnStretch(1, 1)
    for r, a in enumerate(("X", "Y", "Z")):
        lb = QLabel(a); lb.setObjectName("SidebarFieldLabel")
        lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.addWidget(lb, r, 0); l.addWidget(inputs[r], r, 1)
    return l


# ----------------------------------------------------------------------
# 血管面板 — 入口/出口预设 + 沿中心线滑动 seed + 透明度
# ----------------------------------------------------------------------

class VascularPanel(QFrame):
    opacity_changed = pyqtSignal(float)
    entry_preset_changed = pyqtSignal(object)              # 预设 dict
    exit_preset_changed = pyqtSignal(object)
    entry_seed_step = pyqtSignal(int)                      # +N 前进 / -N 后退
    exit_seed_step = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._suspend = False
        self.entry_preset_selector = QComboBox()
        self.exit_preset_selector = QComboBox()
        self.opacity_input = FocusWheelDoubleSpinBox()
        self.opacity_input.setRange(0.2, 1.0); self.opacity_input.setDecimals(2)
        self.opacity_input.setSingleStep(0.05); self.opacity_input.setValue(0.5)
        self.opacity_input.valueChanged.connect(self._emit_opacity)

        self.entry_back = make_step_button("←", "Move entry back one centerline step")
        self.entry_fwd = make_step_button("→", "Move entry forward one centerline step")
        self.entry_back_hold = make_hold_button("◀◀", "Hold to move entry backward")
        self.entry_fwd_hold = make_hold_button("▶▶", "Hold to move entry forward")
        self.exit_back = make_step_button("←", "Move exit back one centerline step")
        self.exit_fwd = make_step_button("→", "Move exit forward one centerline step")
        self.exit_back_hold = make_hold_button("◀◀", "Hold to move exit backward")
        self.exit_fwd_hold = make_hold_button("▶▶", "Hold to move exit forward")
        self.entry_back.clicked.connect(lambda: self.entry_seed_step.emit(-3))
        self.entry_fwd.clicked.connect(lambda: self.entry_seed_step.emit(3))
        self.exit_back.clicked.connect(lambda: self.exit_seed_step.emit(-3))
        self.exit_fwd.clicked.connect(lambda: self.exit_seed_step.emit(3))
        HoldToStepEmitter(self.entry_back_hold, lambda: self.entry_seed_step.emit(-3))
        HoldToStepEmitter(self.entry_fwd_hold, lambda: self.entry_seed_step.emit(3))
        HoldToStepEmitter(self.exit_back_hold, lambda: self.exit_seed_step.emit(-3))
        HoldToStepEmitter(self.exit_fwd_hold, lambda: self.exit_seed_step.emit(3))

        self.entry_preset_selector.currentIndexChanged.connect(self._on_entry_preset)
        self.exit_preset_selector.currentIndexChanged.connect(self._on_exit_preset)

        l = QVBoxLayout(self); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)
        l.addWidget(make_section_label("Entry Preset"))
        l.addWidget(self.entry_preset_selector)
        l.addLayout(self._make_step_row(
            self.entry_back, self.entry_fwd, self.entry_back_hold, self.entry_fwd_hold))
        l.addWidget(make_section_label("Exit Preset"))
        l.addWidget(self.exit_preset_selector)
        l.addLayout(self._make_step_row(
            self.exit_back, self.exit_fwd, self.exit_back_hold, self.exit_fwd_hold))
        l.addWidget(make_section_label("Vessel Opacity"))
        l.addWidget(self.opacity_input)

    def _make_step_row(self, *buttons):
        h = QHBoxLayout(); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        for btn in buttons:
            h.addWidget(btn)
        h.addStretch(1)
        return h

    def set_presets(self, entry_presets, exit_presets, default_entry_label, default_exit_label):
        self._suspend = True
        self.entry_preset_selector.clear()
        for p in entry_presets:
            self.entry_preset_selector.addItem(p["label"], p)
        for i in range(self.entry_preset_selector.count()):
            if self.entry_preset_selector.itemData(i)["label"] == default_entry_label:
                self.entry_preset_selector.setCurrentIndex(i); break
        self.exit_preset_selector.clear()
        for p in exit_presets:
            self.exit_preset_selector.addItem(p["label"], p)
        for i in range(self.exit_preset_selector.count()):
            if self.exit_preset_selector.itemData(i)["label"] == default_exit_label:
                self.exit_preset_selector.setCurrentIndex(i); break
        self._suspend = False

    def set_opacity(self, value):
        self._suspend = True
        self.opacity_input.setValue(float(value))
        self._suspend = False

    def _emit_opacity(self, v):
        if self._suspend: return
        self.opacity_changed.emit(float(v))

    def _on_entry_preset(self):
        if self._suspend: return
        p = self.entry_preset_selector.currentData()
        if p: self.entry_preset_changed.emit(p)

    def _on_exit_preset(self):
        if self._suspend: return
        p = self.exit_preset_selector.currentData()
        if p: self.exit_preset_changed.emit(p)


# ----------------------------------------------------------------------
# 机器人面板 — 三维坐标 + 旋转
# ----------------------------------------------------------------------

class RobotPanel(QFrame):
    transform_changed = pyqtSignal(tuple, tuple)   # position, rotation

    def __init__(self):
        super().__init__()
        self._suspend = False
        self.position_inputs = make_vector_inputs(-5000.0, 5000.0, 1.0, self._emit)
        self.rotation_inputs = make_vector_inputs(-360.0, 360.0, 1.0, self._emit)

        l = QVBoxLayout(self); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)
        l.addWidget(make_section_label("3D Position"))
        l.addLayout(make_vector_layout(self.position_inputs))
        l.addWidget(make_section_label("3-Axis Rotation"))
        l.addLayout(make_vector_layout(self.rotation_inputs))

    def set_transform(self, position, rotation):
        self._suspend = True
        for v, s in zip(position, self.position_inputs): s.setValue(v)
        for v, s in zip(rotation, self.rotation_inputs): s.setValue(v)
        self._suspend = False

    def _emit(self):
        if self._suspend: return
        p = tuple(s.value() for s in self.position_inputs)
        r = tuple(s.value() for s in self.rotation_inputs)
        self.transform_changed.emit(p, r)


# ----------------------------------------------------------------------
# 中心线面板 — tube 粗细 / 可见性 / 分支显示
# ----------------------------------------------------------------------

class CenterlinePanel(QFrame):
    settings_changed = pyqtSignal(float, bool, bool)  # radius, visible, show_all_branches

    def __init__(self):
        super().__init__()
        self._suspend = False
        self.radius_input = FocusWheelDoubleSpinBox()
        self.radius_input.setRange(0.05, 2.0); self.radius_input.setDecimals(2)
        self.radius_input.setSingleStep(0.05); self.radius_input.setValue(0.10)
        self.radius_input.valueChanged.connect(self._emit)
        self.visible_check = QCheckBox("Show centerline"); self.visible_check.setChecked(True)
        self.visible_check.toggled.connect(self._emit)
        self.branches_check = QCheckBox("Show all branches (gray)"); self.branches_check.setChecked(True)
        self.branches_check.toggled.connect(self._emit)

        l = QVBoxLayout(self); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)
        l.addWidget(make_section_label("Line Thickness (tube radius)"))
        l.addWidget(self.radius_input)
        l.addWidget(self.visible_check)
        l.addWidget(self.branches_check)

    def set_settings(self, radius, visible, show_all_branches):
        self._suspend = True
        self.radius_input.setValue(float(radius))
        self.visible_check.setChecked(bool(visible))
        self.branches_check.setChecked(bool(show_all_branches))
        self._suspend = False

    def _emit(self, *_):
        if self._suspend: return
        self.settings_changed.emit(
            float(self.radius_input.value()),
            bool(self.visible_check.isChecked()),
            bool(self.branches_check.isChecked()),
        )


# ----------------------------------------------------------------------
# 中心线导丝面板 — 粗细 + 推进/后退
# ----------------------------------------------------------------------

class GuidewirePanel(QFrame):
    radius_changed = pyqtSignal(float)
    step_requested = pyqtSignal(int)            # +N 前进 / -N 后退
    reset_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._suspend = False
        self.radius_input = FocusWheelDoubleSpinBox()
        self.radius_input.setRange(0.01, 1.0); self.radius_input.setDecimals(3)
        self.radius_input.setSingleStep(0.01); self.radius_input.setValue(0.03)
        self.radius_input.valueChanged.connect(self._emit_radius)

        self.back_btn = make_step_button("← Back 1 Step", "Move back one step")
        self.fwd_btn = make_step_button("Forward 1 Step →", "Move forward one step")
        self.back_hold = make_hold_button("◀◀ Continuous Back", "Hold to move backward")
        self.fwd_hold = make_hold_button("Continuous Forward ▶▶", "Hold to move forward")
        self.reset_btn = make_step_button("⟲ Reset", "Return the guidewire to the entry")
        self.back_btn.clicked.connect(lambda: self.step_requested.emit(-3))
        self.fwd_btn.clicked.connect(lambda: self.step_requested.emit(3))
        HoldToStepEmitter(self.back_hold, lambda: self.step_requested.emit(-3))
        HoldToStepEmitter(self.fwd_hold, lambda: self.step_requested.emit(3))
        self.reset_btn.clicked.connect(self.reset_requested.emit)

        self.progress_label = QLabel("Progress — / —")
        self.progress_label.setObjectName("SidebarFieldLabel")

        l = QVBoxLayout(self); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(8)
        l.addWidget(make_section_label("Guidewire Thickness (tube radius)"))
        l.addWidget(self.radius_input)
        l.addWidget(make_section_label("Single-Step Motion"))
        row1 = QHBoxLayout(); row1.setContentsMargins(0, 0, 0, 0); row1.setSpacing(6)
        row1.addWidget(self.back_btn); row1.addWidget(self.fwd_btn); row1.addStretch(1)
        l.addLayout(row1)
        l.addWidget(make_section_label("Continuous Motion (hold)"))
        row2 = QHBoxLayout(); row2.setContentsMargins(0, 0, 0, 0); row2.setSpacing(6)
        row2.addWidget(self.back_hold); row2.addWidget(self.fwd_hold); row2.addStretch(1)
        l.addLayout(row2)
        row3 = QHBoxLayout(); row3.setContentsMargins(0, 0, 0, 0); row3.setSpacing(6)
        row3.addWidget(self.reset_btn); row3.addStretch(1)
        l.addLayout(row3)
        l.addWidget(self.progress_label)

    def set_radius(self, value):
        self._suspend = True
        self.radius_input.setValue(float(value))
        self._suspend = False

    def set_progress(self, current, total):
        self.progress_label.setText(f"Progress {current} / {total}")

    def _emit_radius(self, v):
        if self._suspend: return
        self.radius_changed.emit(float(v))


# ----------------------------------------------------------------------
# 主侧边栏
# ----------------------------------------------------------------------

class CollapsibleSidebar(QFrame):
    """实验控制侧边栏 — 按选中对象切换控制面板。"""

    # 兼容旧 workspace_page 的信号
    toggled = pyqtSignal(bool)
    transform_changed = pyqtSignal(int, str, tuple, tuple, float)  # scene, model, pos, rot, scale
    vascular_settings_changed = pyqtSignal(int, float, tuple, tuple, float)  # scene, opacity, entry, exit, marker_radius
    entry_preset_selected = pyqtSignal(int, tuple)
    sync_current_scene_requested = pyqtSignal(int)

    # 新增信号
    entry_seed_step_requested = pyqtSignal(int, int)               # scene, delta
    exit_seed_step_requested = pyqtSignal(int, int)
    centerline_settings_changed = pyqtSignal(int, float, bool, bool)  # scene, radius, visible, show_all
    guidewire_radius_changed = pyqtSignal(int, float)
    guidewire_step_requested = pyqtSignal(int, int)                # scene, delta
    guidewire_reset_requested = pyqtSignal(int)

    # 导丝菜单(独立 sidebar 模式)信号
    guidewire_color_changed = pyqtSignal(int, str)
    guidewire_hold_forward_pressed = pyqtSignal(int)
    guidewire_hold_backward_pressed = pyqtSignal(int)
    guidewire_hold_released = pyqtSignal(int)
    inset_toggle_requested = pyqtSignal(int, bool)
    minimap_toggle_requested = pyqtSignal(int, bool)
    guidewire_rotate_cw_requested = pyqtSignal(int)
    guidewire_rotate_ccw_requested = pyqtSignal(int)


    MODEL_ITEMS = [
            ("Vessel Model", "vascular"),
            ("Robot Model", "robot"),
            ("Centerline", "centerline"),
    ]
    DEFAULT_MARKER_RADIUS = 0.3
    FIXED_MODEL_SCALE = 0.1

    def __init__(self, compact_vascular_controls=True):
        super().__init__()
        self.compact_vascular_controls = compact_vascular_controls
        self.expanded = False
        self.expanded_width = 320
        self._suspend_emit = False
        # per-scene state
        self.robot_transforms = {}             # scene_idx -> (pos, rot)
        self.vascular_states = {}              # scene_idx -> {opacity, entry_label, exit_label}
        self.centerline_states = {}            # scene_idx -> {radius, visible, show_all}
        self.guidewire_states = {}             # scene_idx -> {radius}
        self.guidewire_progress = {}           # scene_idx -> (current, total)
        self.entry_presets = []
        self.exit_presets = []
        self.setObjectName("WorkbenchSidebar")
        self.setStyleSheet(SIDEBAR_STYLE)
        self.build_ui()
        self.apply_state()

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    def build_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(14, 14, 14, 14)
        self.layout.setSpacing(10)

        self.title_label = QLabel("Experiment Controls"); self.title_label.setObjectName("SidebarTitle")
        self.title_label.setWordWrap(True)
        self.sync_to_other_scenes_button = QToolButton()
        self.sync_to_other_scenes_button.setObjectName("SidebarSyncButton")
        self.sync_to_other_scenes_button.setText("Sync"); self.sync_to_other_scenes_button.setCheckable(True)
        self.sync_to_other_scenes_button.setChecked(True)
        self.sync_to_other_scenes_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sync_to_other_scenes_button.setToolTip(
            "Apply current adjustments to the other experiment scenes"
        )
        self.sync_to_other_scenes_button.toggled.connect(self._on_sync_toggled)

        self.active_scene_label = QLabel("Current: Experiment Scene 1")
        self.active_scene_label.setObjectName("SidebarActiveScene")
        self.active_scene_label.setWordWrap(True)
        title_row = QHBoxLayout(); title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addWidget(self.title_label); title_row.addStretch(1)
        title_row.addWidget(self.sync_to_other_scenes_button)

        self.hint_label = QLabel("Select a scene and object to adjust"); self.hint_label.setObjectName("SidebarHint")
        self.hint_label.setWordWrap(True)

        self.scene_selector = QComboBox()
        for index in range(4): self.scene_selector.addItem(f"Experiment Scene {index + 1}", index)
        self.scene_selector.currentIndexChanged.connect(self._on_scene_changed)
        # 标记单场景模式 — 由 workspace.toggle_scene_mode 调用 set_single_scene_mode 切换
        self._single_scene_mode = False

        self.model_selector = QComboBox()
        for label, key in self.MODEL_ITEMS: self.model_selector.addItem(label, key)
        self.model_selector.currentIndexChanged.connect(self._on_model_changed)

        # 4 个对象面板
        self.vascular_panel = VascularPanel()
        self.robot_panel = RobotPanel()
        self.centerline_panel = CenterlinePanel()
        self.guidewire_panel = GuidewirePanel()

        self.vascular_panel.opacity_changed.connect(self._on_vascular_opacity)
        self.vascular_panel.entry_preset_changed.connect(self._on_entry_preset)
        self.vascular_panel.exit_preset_changed.connect(self._on_exit_preset)
        self.vascular_panel.entry_seed_step.connect(
            lambda d: self.entry_seed_step_requested.emit(self.current_scene_index(), d))
        self.vascular_panel.exit_seed_step.connect(
            lambda d: self.exit_seed_step_requested.emit(self.current_scene_index(), d))

        self.robot_panel.transform_changed.connect(self._on_robot_transform)

        self.centerline_panel.settings_changed.connect(
            lambda r, v, s: self._emit_centerline(r, v, s))

        self.guidewire_panel.radius_changed.connect(
            lambda r: self.guidewire_radius_changed.emit(self.current_scene_index(), r))
        self.guidewire_panel.step_requested.connect(
            lambda d: self.guidewire_step_requested.emit(self.current_scene_index(), d))
        self.guidewire_panel.reset_requested.connect(
            lambda: self.guidewire_reset_requested.emit(self.current_scene_index()))

        self.panel_stack = QStackedWidget()
        self.panel_stack.addWidget(self.vascular_panel)
        self.panel_stack.addWidget(self.robot_panel)
        self.panel_stack.addWidget(self.centerline_panel)
        self.panel_stack.addWidget(self.guidewire_panel)

        self.scroll_content = QFrame(); self.scroll_content.setObjectName("SidebarScrollContent")
        scroll_layout = QVBoxLayout(self.scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0); scroll_layout.setSpacing(10)
        scroll_layout.addWidget(self.hint_label)
        scroll_layout.addWidget(make_section_label("Experiment Scene"))
        scroll_layout.addWidget(self.scene_selector)
        scroll_layout.addWidget(make_section_label("Object"))
        scroll_layout.addWidget(self.model_selector)
        scroll_layout.addWidget(self.panel_stack)
        scroll_layout.addStretch(1)

        self.scroll_area = QScrollArea(); self.scroll_area.setObjectName("SidebarScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidget(self.scroll_content)

        # ----- 导丝菜单页（与实验控制共用 sidebar，由顶部栏按钮切换） -----
        from zmd.ui.panels.guidewire_menu_panel import GuidewireMenuPanel
        self.guidewire_menu_panel = GuidewireMenuPanel()
        self._wire_guidewire_menu()

        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self.scroll_area)            # page 0: 实验控制
        self.mode_stack.addWidget(self.guidewire_menu_panel)   # page 1: 导丝菜单
        self._current_mode = "exp"

        self.layout.addLayout(title_row)
        self.layout.addWidget(self.active_scene_label)
        self.layout.addWidget(self.mode_stack, 1)
        self._refresh_current_panel()

    def set_mode(self, mode: str):
        mode = "guidewire" if mode == "guidewire" else "exp"
        self._current_mode = mode
        self.mode_stack.setCurrentIndex(0 if mode == "exp" else 1)
        self.title_label.setText(
            "Experiment Controls" if mode == "exp" else "Guidewire"
        )

    def current_mode(self) -> str:
        return self._current_mode

    def _wire_guidewire_menu(self):
        p = self.guidewire_menu_panel
        sf = lambda: self.current_scene_index()
        p.radius_changed.connect(lambda r: self.guidewire_radius_changed.emit(sf() or 0, r))
        p.color_changed.connect(lambda c: self.guidewire_color_changed.emit(sf() or 0, c))
        p.advance_clicked.connect(lambda: self.guidewire_step_requested.emit(sf() or 0, 3))
        p.retract_clicked.connect(lambda: self.guidewire_step_requested.emit(sf() or 0, -3))
        p.hold_forward_pressed.connect(lambda: self.guidewire_hold_forward_pressed.emit(sf() or 0))
        p.hold_backward_pressed.connect(lambda: self.guidewire_hold_backward_pressed.emit(sf() or 0))
        p.hold_released.connect(lambda: self.guidewire_hold_released.emit(sf() or 0))
        p.reset_requested.connect(lambda: self.guidewire_reset_requested.emit(sf() or 0))
        p.inset_toggle_changed.connect(lambda v: self.inset_toggle_requested.emit(sf() or 0, v))
        p.minimap_toggle_changed.connect(lambda v: self.minimap_toggle_requested.emit(sf() or 0, v))
        p.rotate_cw_clicked.connect(lambda: self.guidewire_rotate_cw_requested.emit(sf() or 0))
        p.rotate_ccw_clicked.connect(lambda: self.guidewire_rotate_ccw_requested.emit(sf() or 0))

    # ------------------------------------------------------------------
    # 折叠 / 展开
    # ------------------------------------------------------------------

    def toggle(self):
        self.expanded = not self.expanded
        self.apply_state()
        self.toggled.emit(self.expanded)

    def apply_state(self):
        self.setVisible(self.expanded)
        self.setFixedWidth(self.expanded_width if self.expanded else 0)

    # ------------------------------------------------------------------
    # 状态访问
    # ------------------------------------------------------------------

    def current_scene_index(self):
        return self.scene_selector.currentData()

    def set_current_scene_index(self, scene_index):
        for i in range(self.scene_selector.count()):
            if self.scene_selector.itemData(i) == scene_index:
                self.scene_selector.setCurrentIndex(i)
                self._refresh_active_scene_label()
                return

    def set_single_scene_mode(self, single: bool):
        """单场景模式:scene_selector 只显示场景 1 + 隐藏"同步"按钮。"""
        single = bool(single)
        self._single_scene_mode = single
        # 强制切到 scene 0
        self.set_current_scene_index(0)
        # 隐藏/恢复 scene 2-4 选项
        model = self.scene_selector.model()
        for i in range(self.scene_selector.count()):
            data = self.scene_selector.itemData(i)
            visible = (not single) or (data == 0)
            self.scene_selector.view().setRowHidden(i, not visible)
            item = model.item(i)
            if item is not None:
                item.setEnabled(visible)
        # 同步按钮 + 标题在单场景下没意义
        self.sync_to_other_scenes_button.setVisible(not single)
        self.active_scene_label.setVisible(not single)

    def _refresh_active_scene_label(self):
        idx = self.current_scene_index()
        if idx is None:
            self.active_scene_label.setText("Current: —")
        else:
            self.active_scene_label.setText(f"Current: Experiment Scene {idx + 1}")

    def current_model_key(self):
        return self.model_selector.currentData()

    def sync_to_other_scenes_enabled(self):
        return self.sync_to_other_scenes_button.isChecked()

    # ------------------------------------------------------------------
    # 外部数据注入 (workspace_page 使用)
    # ------------------------------------------------------------------

    def set_vascular_presets(self, entry_presets, exit_presets,
                             default_entry_label=None, default_exit_label=None):
        self.entry_presets = list(entry_presets)
        self.exit_presets = list(exit_presets)
        self.vascular_panel.set_presets(
            self.entry_presets, self.exit_presets,
            default_entry_label, default_exit_label,
        )

    def set_transform(self, scene_index, model_key, position, rotation, scale=None):
        """兼容老接口，仅对机器人保留 transform；血管 transform 忽略（已废除）。"""
        if model_key == "robot":
            self.robot_transforms[scene_index] = (tuple(position), tuple(rotation))
            if scene_index == self.current_scene_index() and self.current_model_key() == "robot":
                self.robot_panel.set_transform(*self.robot_transforms[scene_index])

    def set_vascular_settings(self, scene_index, opacity, entry_point, exit_point,
                              marker_radius=DEFAULT_MARKER_RADIUS):
        """记录场景的血管设置。entry/exit 坐标由 workspace 通过预设/seed-step 维护，这里只关心
        opacity 与预设标签。entry/exit/marker_radius 参数保留以兼容老调用。"""
        del entry_point, exit_point, marker_radius
        state = self.vascular_states.setdefault(scene_index, {})
        state["opacity"] = float(opacity)
        if scene_index == self.current_scene_index() and self.current_model_key() == "vascular":
            self.vascular_panel.set_opacity(state["opacity"])

    def set_centerline_settings(self, scene_index, radius, visible, show_all_branches):
        self.centerline_states[scene_index] = {
            "radius": float(radius), "visible": bool(visible),
            "show_all": bool(show_all_branches),
        }
        if scene_index == self.current_scene_index() and self.current_model_key() == "centerline":
            self.centerline_panel.set_settings(radius, visible, show_all_branches)

    def set_guidewire_settings(self, scene_index, radius):
        self.guidewire_states[scene_index] = {"radius": float(radius)}
        if scene_index == self.current_scene_index() and self.current_model_key() == "guidewire":
            self.guidewire_panel.set_radius(radius)

    def set_guidewire_progress(self, scene_index, current, total):
        self.guidewire_progress[scene_index] = (int(current), int(total))
        if scene_index == self.current_scene_index() and self.current_model_key() == "guidewire":
            self.guidewire_panel.set_progress(current, total)

    # ------------------------------------------------------------------
    # 切换 selector
    # ------------------------------------------------------------------

    def _on_scene_changed(self):
        self._refresh_current_panel()
        self._refresh_active_scene_label()

    def _on_model_changed(self):
        idx = self.model_selector.currentIndex()
        self.panel_stack.setCurrentIndex(idx)
        self._refresh_current_panel()

    def _refresh_current_panel(self):
        """根据当前 (scene, model) 把对应面板的控件值灌入。"""
        scene = self.current_scene_index()
        if scene is None: return
        model = self.current_model_key()
        self._suspend_emit = True
        try:
            if model == "vascular":
                state = self.vascular_states.get(scene, {})
                self.vascular_panel.set_opacity(state.get("opacity", 0.5))
            elif model == "robot":
                pos, rot = self.robot_transforms.get(scene, ((0, 0, 0), (0, 0, 0)))
                self.robot_panel.set_transform(pos, rot)
            elif model == "centerline":
                s = self.centerline_states.get(scene, {"radius": 0.15, "visible": True, "show_all": True})
                self.centerline_panel.set_settings(s["radius"], s["visible"], s["show_all"])
            elif model == "guidewire":
                s = self.guidewire_states.get(scene, {"radius": 0.20})
                self.guidewire_panel.set_radius(s["radius"])
                cur, total = self.guidewire_progress.get(scene, (0, 0))
                self.guidewire_panel.set_progress(cur, total)
        finally:
            self._suspend_emit = False

    # ------------------------------------------------------------------
    # 信号转发
    # ------------------------------------------------------------------

    def _on_vascular_opacity(self, opacity):
        if self._suspend_emit: return
        scene = self.current_scene_index()
        state = self.vascular_states.setdefault(scene, {})
        state["opacity"] = float(opacity)
        # 旧 vascular_settings_changed 签名向后兼容：entry/exit 用占位元组，marker_radius 用默认值
        self.vascular_settings_changed.emit(
            scene, float(opacity), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), self.DEFAULT_MARKER_RADIUS
        )

    def _on_entry_preset(self, preset):
        if self._suspend_emit: return
        scene = self.current_scene_index()
        robot_pos = tuple(preset.get("robot_position", (0.0, 0.0, 0.0)))
        self.entry_preset_selected.emit(scene, robot_pos)

    def _on_exit_preset(self, preset):
        if self._suspend_emit: return
        # 出口预设选择 — workspace_page 监听 vascular_settings_changed 处理；此处不额外发信号
        scene = self.current_scene_index()
        # 用专门信号转发？复用 entry_preset_selected 不合适。直接借 vascular_settings_changed
        # 但需要明确语义 — workspace 自行 wire。
        # 简化：用 entry_preset_selected 一类用法，由 workspace 在 emit 后查 selector 当前值
        self.exit_preset_changed_emit(preset)

    def exit_preset_changed_emit(self, preset):
        """供外部连接的辅助信号入口。"""
        # 复用 vascular_settings_changed？语义不对。暴露给 workspace 自己读 current_exit_preset()
        # 触发一次空 vascular_settings_changed 让 workspace 重新挑路由
        # 简单方案：workspace 监听 sidebar.vascular_panel.exit_preset_changed
        # 这里转发到 workspace 直接连接的 sidebar.vascular_panel.exit_preset_changed 即可。
        pass

    def _on_robot_transform(self, position, rotation):
        if self._suspend_emit: return
        scene = self.current_scene_index()
        self.robot_transforms[scene] = (position, rotation)
        self.transform_changed.emit(scene, "robot", position, rotation, self.FIXED_MODEL_SCALE)

    def _emit_centerline(self, radius, visible, show_all):
        if self._suspend_emit: return
        scene = self.current_scene_index()
        self.centerline_states[scene] = {
            "radius": float(radius), "visible": bool(visible), "show_all": bool(show_all),
        }
        self.centerline_settings_changed.emit(scene, float(radius), bool(visible), bool(show_all))

    def _on_sync_toggled(self, checked):
        if not checked or self._suspend_emit: return
        self.sync_current_scene_requested.emit(self.current_scene_index())

    # ------------------------------------------------------------------
    # 公开 - 给 workspace_page 直接拿到 panel 来连接信号
    # ------------------------------------------------------------------

    @property
    def exit_preset_changed(self):
        return self.vascular_panel.exit_preset_changed
