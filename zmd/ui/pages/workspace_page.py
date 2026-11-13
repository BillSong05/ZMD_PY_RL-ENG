# 职责：通用工作台页面，根据 ProjectConfig 加载模型并展示 4 个独立实验场景。
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QIcon, QLinearGradient, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractButton, QApplication, QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QToolButton, QVBoxLayout,
    QWidget,
)

from zmd.io.project_loader import ProjectConfig
from zmd.scene.scene_config import (
    DEFAULT_CENTERLINE_SETTINGS,
    DEFAULT_ENTRY_POINT_PRESETS,
    DEFAULT_EXIT_POINT_PRESETS,
    DEFAULT_GUIDEWIRE_SETTINGS,
    DEFAULT_MODEL_TRANSFORMS,
    DEFAULT_VASCULAR_SETTINGS,
    SCENE_LAYOUTS,
)
import numpy as np
from zmd.centerline.centerline_model import Centerline, pick_route
from zmd.scene.scene_manager import SceneManager
from zmd.rendering.pyvista_view import Scene3DWidget
from zmd.simulation.simulation_engine import SimulationEngine
from zmd.simulation.backends.dummy_backend import DummyPhysicsBackend
from zmd.simulation.guidewire import MujocoGuidewire
from zmd.domain.guidewire_base import GuidewireAction, GuidewireBase
from zmd.ui.panels.control_sidebar import SIDEBAR_STYLE, CollapsibleSidebar
from zmd.ui.panels.debug_panel import WorkbenchDebugPanel
from zmd.ui.widgets import WINDOW_CONTROLS_STYLE, WindowControls
from zmd.ui.widgets.guidewire_inset import GuidewireInsetView
from zmd.ui.widgets.minimap import CenterlineMinimap


DESIGN_WIDTH = 1360
DESIGN_HEIGHT = 860


WORKBENCH_STYLE = f"""
QWidget {{
    background: transparent;
    color: #eef6ff;
    font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
}}
{WINDOW_CONTROLS_STYLE}
{SIDEBAR_STYLE}
#HomeButton, #SceneEditButton, #DebugInfoButton, #GuidewireMenuButton, #SimStepButton {{
    background: rgba(15, 23, 42, 0.42);
    border: 1px solid rgba(148, 163, 184, 0.38);
    border-radius: 16px;
    color: #dbe7f6;
    font-size: 15px;
    font-weight: 800;
    padding: 7px 14px;
}}
#HomeButton:hover, #SceneEditButton:hover, #DebugInfoButton:hover, #GuidewireMenuButton:hover, #SimStepButton:hover {{
    background: rgba(59, 130, 246, 0.72);
    border-color: rgba(147, 197, 253, 0.86);
    color: #ffffff;
}}
#ViewportPanel {{
    background: rgba(10, 35, 80, 0.48);
    border: 1px solid rgba(59, 130, 246, 0.48);
    border-radius: 16px;
}}
#ViewportPanel[activeControl="true"] {{
    border: 2px solid rgba(34, 197, 94, 0.95);
}}
#ViewportProgress {{
    background: rgba(8, 17, 32, 0.55);
    color: #dbe7f6;
    border: 1px solid rgba(96, 165, 250, 0.45);
    border-radius: 10px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 700;
}}
#ViewportToolButton {{
    background: rgba(15, 23, 42, 0.48);
    border: 1px solid rgba(148, 163, 184, 0.36);
    border-radius: 9px;
    color: #dbe7f6;
    font-size: 12px;
    font-weight: 800;
    min-width: 26px;
    min-height: 22px;
    padding: 2px 5px;
}}
#ViewportToolButton:hover {{
    background: rgba(59, 130, 246, 0.72);
    border-color: rgba(147, 197, 253, 0.86);
    color: #ffffff;
}}
#ViewportToolButton:checked {{
    background: rgba(244, 114, 182, 0.82);
    border-color: rgba(251, 207, 232, 0.95);
    color: #ffffff;
}}
#ViewportTitle {{
    color: #dbeafe;
    font-size: 14px;
    font-weight: 800;
}}
"""


class WorkspacePage(QWidget):
    """通用工作台页面，根据 ProjectConfig 加载模型。默认工程和自定义工程统一使用此类。"""
    close_requested = pyqtSignal()
    minimize_requested = pyqtSignal()
    fullscreen_requested = pyqtSignal()
    return_home_requested = pyqtSignal()

    def __init__(self, project_config: ProjectConfig, parent=None):
        super().__init__(parent)
        self.project_config = project_config
        self.drag_position = None
        self.content = WorkspaceContent(project_config)
        self.setAutoFillBackground(True)
        self.setStyleSheet(WORKBENCH_STYLE)
        self.build_ui()

        self.content.close_requested.connect(self.close_requested.emit)
        self.content.minimize_requested.connect(self.minimize_requested.emit)
        self.content.fullscreen_requested.connect(self.fullscreen_requested.emit)
        self.content.return_home_requested.connect(self.return_home_requested.emit)

    def set_fullscreen_state(self, is_fullscreen):
        self.content.set_fullscreen_state(is_fullscreen)

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.content)

    def shutdown(self):
        if self.content is not None:
            self.content.shutdown()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0.0, 0.0, float(self.width()), float(self.height()))
        gradient.setColorAt(0.0, QColor("#061124"))
        gradient.setColorAt(0.55, QColor("#07101f"))
        gradient.setColorAt(1.0, QColor("#091827"))
        painter.fillRect(self.rect(), QBrush(gradient))

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self.window().isFullScreen()
            and not self.is_interactive_position(event.position().toPoint())
        ):
            self.drag_position = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_position is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_position is not None:
            self.drag_position = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def is_interactive_position(self, position):
        widget = self.childAt(position)
        while widget:
            if isinstance(widget, QAbstractButton) or isinstance(widget, Scene3DWidget):
                return True
            if widget.objectName() in {"DebugPanel", "DebugPanelLog"}:
                return True
            widget = widget.parentWidget()
        return False


class WorkspaceContent(QWidget):
    close_requested = pyqtSignal()
    minimize_requested = pyqtSignal()
    fullscreen_requested = pyqtSignal()
    return_home_requested = pyqtSignal()

    def __init__(self, project_config: ProjectConfig):
        super().__init__()
        self.project_config = project_config
        self.compact_vascular_controls = project_config.compact_vascular_controls

        self.window_controls = WindowControls()
        self.sidebar = CollapsibleSidebar(compact_vascular_controls=self.compact_vascular_controls)
        self.sidebar.set_vascular_presets(
            DEFAULT_ENTRY_POINT_PRESETS,
            DEFAULT_EXIT_POINT_PRESETS,
            default_entry_label="Entry 1",
            default_exit_label="Exit 3",
        )
        self.experiment_scenes = {}
        self.viewport_panels = {}
        self.viewport_scene_widgets = {}
        self.viewport_positions = {}
        self.viewport_click_targets = {}
        self.viewport_icon_cache = {}
        self.viewport_grid = None
        self.viewport_grid_spacing = 16
        self.zoom_buttons = {}
        self.hide_buttons = {}
        self.progress_overlays = {}      # scene_idx -> QLabel
        self.inset_views = {}            # scene_idx -> GuidewireInsetView
        self.minimaps = {}               # scene_idx -> CenterlineMinimap
        self.minimap_buttons = {}        # scene_idx -> QToolButton (toggle)
        self.active_control_scene_index = None
        # 启动默认单场景:只显示 scene 0,zoomed_scene_index=0 让 grid 把它放大占满
        self.zoomed_scene_index = 0
        self.single_scene_mode = True
        self.scene_edit_button = None
        self.debug_info_button = None
        self.scene_manager = None
        self.simulation_engines: dict = {}       # scene_idx -> SimulationEngine (legacy ref)
        self.active_guidewires: dict = {}        # scene_idx -> GuidewireBase 实例
        # MuJoCo 导丝实例缓存:reset 时不销毁,下次可复用、保留物理状态。
        self._mujoco_gw_cache: dict = {}         # scene_idx -> MujocoGuidewire
        # MuJoCo 导丝当前物理参数(面板"应用并重建"会更新它,新建导丝时用它做初值)
        try:
            from zmd.simulation.guidewire.mujoco_config import MujocoGuidewireConfig
            self._mujoco_config = MujocoGuidewireConfig()
        except Exception:
            self._mujoco_config = None
        self.scene_state: dict = {}              # scene_idx -> {entry_preset, exit_preset, entry_offset, exit_offset, full_branch, trimmed}
        self.vtp_branches = None                 # 完整中心线树（所有 polyline 共享）
        self.follow_buttons = {}                 # scene_idx -> QToolButton
        # 键盘 ↑/↓ 推进/后退导丝 — 90ms 重复
        self._key_step_timer = QTimer(self)
        self._key_step_timer.setInterval(90)
        self._key_step_timer.timeout.connect(self._on_key_step_tick)
        self._key_step_delta = 0
        # 键盘 ←/→ 绕轴旋转导丝 — 同 90ms 重复(长按连续转,模拟介入实验)
        self._key_rotate_timer = QTimer(self)
        self._key_rotate_timer.setInterval(90)
        self._key_rotate_timer.timeout.connect(self._on_key_rotate_tick)
        self._key_rotate_dir = 0   # +1 = cw, -1 = ccw, 0 = idle
        self.debug_panel = WorkbenchDebugPanel()
        self.debug_panel_host = None
        self.setMinimumSize(DESIGN_WIDTH, DESIGN_HEIGHT)
        self.setAutoFillBackground(True)
        self.setStyleSheet(WORKBENCH_STYLE)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.build_ui()
        # 全局键盘事件过滤 — 让 ↑/↓ 在任何子控件（包括 QtInteractor）有焦点时也能捕获
        QApplication.instance().installEventFilter(self)

    def set_fullscreen_state(self, is_fullscreen):
        self.window_controls.set_fullscreen_state(is_fullscreen)

    def build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(28, 18, 28, 28)
        root_layout.setSpacing(0)

        self.window_controls.minimize_requested.connect(self.minimize_requested.emit)
        self.window_controls.fullscreen_requested.connect(self.fullscreen_requested.emit)
        self.window_controls.close_requested.connect(self.close_requested.emit)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        home_button = QPushButton("←  Home")
        home_button.setObjectName("HomeButton")
        home_button.setCursor(Qt.CursorShape.PointingHandCursor)
        home_button.clicked.connect(self.return_home_requested.emit)

        self.scene_edit_button = QPushButton("Scene Editor")
        self.scene_edit_button.setObjectName("SceneEditButton")
        self.scene_edit_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.scene_edit_button.clicked.connect(lambda: self.toggle_sidebar_mode("exp"))

        self.debug_info_button = QPushButton("Debug Info")
        self.debug_info_button.setObjectName("DebugInfoButton")
        self.debug_info_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.debug_info_button.clicked.connect(self.toggle_debug_panel)

        self.guidewire_menu_button = QPushButton("Guidewire")
        self.guidewire_menu_button.setObjectName("GuidewireMenuButton")
        self.guidewire_menu_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.guidewire_menu_button.clicked.connect(lambda: self.toggle_sidebar_mode("guidewire"))

        # 强化学习入口 — 切换右侧 RL 占位面板的显隐(收缩式)
        self.rl_button = QPushButton("Reinforcement Learning")
        self.rl_button.setObjectName("GuidewireMenuButton")  # 共用样式
        self.rl_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.rl_button.setToolTip("Show or hide the reinforcement learning panel")
        self.rl_button.clicked.connect(self.toggle_rl_panel)

        # 单场景 / 多场景 切换 — 当前阶段多场景功能未启用,按钮保留 + 置灰 +
        # tooltip 说明,等多场景训练对比逻辑接入后再 setEnabled(True)。
        self.scene_mode_button = QPushButton("Multi-Scene")
        self.scene_mode_button.setObjectName("GuidewireMenuButton")  # 共用样式
        self.scene_mode_button.setCursor(Qt.CursorShape.ForbiddenCursor)
        self.scene_mode_button.setEnabled(False)
        self.scene_mode_button.setToolTip("Multi-scene mode is not available yet")
        self.scene_mode_button.clicked.connect(self.toggle_scene_mode)

        top_bar.addWidget(home_button)
        top_bar.addWidget(self.scene_edit_button)
        top_bar.addWidget(self.debug_info_button)
        top_bar.addWidget(self.guidewire_menu_button)
        top_bar.addWidget(self.rl_button)
        top_bar.addStretch(1)
        top_bar.addWidget(self.scene_mode_button)
        top_bar.addWidget(self.window_controls)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)
        self.sidebar.transform_changed.connect(self.apply_model_transform)
        self.sidebar.vascular_settings_changed.connect(self.apply_vascular_settings)
        self.sidebar.entry_preset_selected.connect(self.apply_entry_preset_robot_position)
        self.sidebar.sync_current_scene_requested.connect(self.sync_current_scene_to_others)
        self.sidebar.vascular_panel.entry_preset_changed.connect(self.apply_entry_preset_route)
        self.sidebar.exit_preset_changed.connect(self.apply_exit_preset_route)
        self.sidebar.entry_seed_step_requested.connect(self.step_entry_seed)
        self.sidebar.exit_seed_step_requested.connect(self.step_exit_seed)
        self.sidebar.centerline_settings_changed.connect(self.apply_centerline_settings)
        self.sidebar.guidewire_radius_changed.connect(self.apply_guidewire_radius)
        self.sidebar.guidewire_step_requested.connect(self.step_guidewire)
        self.sidebar.guidewire_reset_requested.connect(self.reset_guidewire)
        # 导丝菜单专属信号
        self.sidebar.guidewire_hold_forward_pressed.connect(self._on_hold_forward)
        self.sidebar.guidewire_hold_backward_pressed.connect(self._on_hold_backward)
        self.sidebar.guidewire_hold_released.connect(self._on_hold_released)
        self.sidebar.guidewire_color_changed.connect(self._on_guidewire_color)
        self.sidebar.inset_toggle_requested.connect(self.toggle_follow)
        self.sidebar.minimap_toggle_requested.connect(self.toggle_minimap)
        # RL 占位面板(右侧,与左侧 control_sidebar 对称),默认折叠
        from zmd.ui.panels.rl_panel import RLPanel
        self.rl_panel = RLPanel()
        # 把入口/出口预设的 label 灌给 RL 面板,让推理路线下拉能选
        self.rl_panel.set_route_presets(
            [ep["label"] for ep in DEFAULT_ENTRY_POINT_PRESETS],
            [xp["label"] for xp in DEFAULT_EXIT_POINT_PRESETS],
        )
        self.rl_panel.algorithm_changed.connect(self._on_rl_algorithm_changed)
        self.rl_panel.start_clicked.connect(self._on_rl_start)
        self.rl_panel.stop_clicked.connect(self._on_rl_stop)
        self.rl_panel.load_model_requested.connect(self._on_rl_load_model)
        self.rl_panel.reward_params_changed.connect(self._on_rl_reward_params_changed)
        self.rl_panel.run_model_clicked.connect(self._on_rl_run_model)
        self.rl_panel.stop_model_clicked.connect(self._on_rl_stop_model)
        self._rl_inference = None       # PPOInferenceController(懒构造)
        self._rl_loaded_ckpt = None     # 当前加载的 ckpt 路径

        content_layout.addWidget(self.sidebar)
        content_layout.addLayout(self.build_viewport_grid(), 1)
        content_layout.addWidget(self.rl_panel)

        # 将所有 3D 视窗委托给 SceneManager 管理模型加载（共享 MeshCache 避免 4× STL 重读）
        from zmd.io.mesh_cache import MeshCache
        self.mesh_cache = MeshCache()
        self.scene_manager = SceneManager(self.project_config, self.experiment_scenes,
                                          mesh_cache=self.mesh_cache)
        # 先只立即加载 scene 0；其他 scene 通过 QTimer 分帧延迟
        self.scene_manager.ensure_initialized(0)
        self.active_control_scene_index = 0
        for idx in sorted(self.experiment_scenes.keys()):
            if idx == 0:
                continue
            QTimer.singleShot(150 * idx, lambda i=idx: self._lazy_init_scene(i))

        # 加载 VMTK 中心线（如果存在）
        self._load_vmtk_centerline()

        # 初始化仿真引擎
        self._init_simulation_engine()

        self.debug_panel_host = QWidget()
        debug_panel_layout = QHBoxLayout(self.debug_panel_host)
        debug_panel_layout.setContentsMargins(0, 12, 0, 0)
        debug_panel_layout.setSpacing(0)
        debug_panel_layout.addWidget(self.debug_panel, 1)
        self.debug_panel_host.setVisible(False)
        self.sidebar.toggled.connect(lambda _: self.update_debug_panel_width())

        root_layout.addLayout(top_bar)
        root_layout.addSpacing(18)
        root_layout.addLayout(content_layout, 1)
        root_layout.addWidget(self.debug_panel_host)
        self.update_debug_panel_width()
        # 应用启动默认单场景布局(zoomed_scene_index=0 在 __init__ 设好,这里触发布局
        # + 隐藏 viewport 缩放/隐藏按钮 + debug 面板切单场景标题)
        QTimer.singleShot(0, lambda: (self.apply_viewport_zoom_layout(),
                                      self._apply_single_scene_ui()))

    def shutdown(self):
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        try:
            QApplication.instance().removeEventFilter(self)
        except Exception:
            pass
        for timer_name in ("_key_step_timer", "_key_rotate_timer"):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass
        inference = getattr(self, "_rl_inference", None)
        if inference is not None:
            try:
                inference.stop()
            except Exception:
                pass
            self._rl_inference = None
        trainer = getattr(self, "_rl_trainer", None)
        if trainer is not None:
            try:
                trainer.request_stop()
            except Exception:
                pass
        thread = getattr(self, "_rl_thread", None)
        if thread is not None:
            try:
                thread.quit()
                thread.wait(5000)
            except Exception:
                pass
            self._rl_thread = None
        self._rl_trainer = None
        for inset in list(getattr(self, "inset_views", {}).values()):
            try:
                inset.shutdown()
            except Exception:
                pass
        for scene in list(getattr(self, "viewport_scene_widgets", {}).values()):
            try:
                scene.shutdown()
            except Exception:
                pass

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def toggle_sidebar_mode(self, mode: str):
        """场景编辑 / 导丝菜单 互斥切换：同模式再次点击则折叠；不同模式则切换页并保持展开。"""
        target = "guidewire" if mode == "guidewire" else "exp"
        if self.sidebar.expanded and self.sidebar.current_mode() == target:
            self.sidebar.toggle()              # 收起
        else:
            self.sidebar.set_mode(target)
            if not self.sidebar.expanded:
                self.sidebar.toggle()          # 展开
        self._refresh_sidebar_buttons()

    def toggle_rl_panel(self):
        """切换强化学习占位面板的显隐。"""
        try:
            self.rl_panel.toggle()
            self.rl_button.setText(
                "Hide Reinforcement Learning"
                if self.rl_panel.expanded
                else "Reinforcement Learning"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 强化学习面板回调(当前为 stub,后端接入时把 print 换成实际逻辑)
    # ------------------------------------------------------------------

    def _on_rl_algorithm_changed(self, name: str):
        self.debug_panel.append_message(f"[RL] Algorithm changed → {name}")
        if name in {"PPO", "SAC", "TD3", "DDPG", "DQN"}:
            self.rl_panel.set_status(f"{name} ready")
        else:
            self.rl_panel.set_status(f"{name} is not available")

    def _on_rl_start(self):
        from PyQt6.QtCore import QThread
        algo = self.rl_panel.current_algorithm()
        if algo not in {"PPO", "SAC", "TD3", "DDPG", "DQN"}:
            self.debug_panel.append_message(
                f"[RL] {algo} is not available; select PPO or SAC"
            )
            self.rl_panel.set_training_active(False)
            self.rl_panel.set_status(f"{algo} is not available")
            return

        # 用当前 scene 0 的 centerline + 全局 vessel SDF 跑训练
        scene_index = 0
        st = self.scene_state.get(scene_index)
        if st is None or st.get("trimmed") is None:
            self.debug_panel.append_message("[RL] The current scene is not ready for training")
            self.rl_panel.set_training_active(False)
            self.rl_panel.set_status("Not ready")
            return

        params = self.rl_panel.get_reward_params()
        cfg = self.rl_panel.get_training_config()

        # 泛化训练:枚举所有 (entry × exit) 路线;单路训练:用面板选定的那条
        centerlines = None
        single_cl = None
        if cfg.get("generalize"):
            centerlines = self._build_all_route_centerlines()
            if not centerlines:
                self.debug_panel.append_message(
                    "[RL] Generalized training failed: no routes could be built"
                )
                self.rl_panel.set_training_active(False)
                return
            self.debug_panel.append_message(
                f"[RL] {algo} generalized training started routes={len(centerlines)} "
                f"episodes={cfg['max_episodes']} "
                f"max_steps={cfg['max_steps_per_episode']} seed={cfg['seed']} "
                f"speed={cfg['speed_config']['profile']}"
            )
        else:
            # 单路训练:按面板的 训练起点/终点 构造指定路线(不再固定用 scene 0)
            entry_idx = max(0, min(int(cfg.get("train_entry_idx", 0)),
                                   len(DEFAULT_ENTRY_POINT_PRESETS) - 1))
            exit_idx = max(0, min(int(cfg.get("train_exit_idx", 0)),
                                  len(DEFAULT_EXIT_POINT_PRESETS) - 1))
            ep = DEFAULT_ENTRY_POINT_PRESETS[entry_idx]
            xp = DEFAULT_EXIT_POINT_PRESETS[exit_idx]
            single_cl = self._build_route_centerline(ep, xp)
            if single_cl is None:
                self.debug_panel.append_message(
                    f"[RL] Single-route training failed: "
                    f"{ep['label']}→{xp['label']} could not be built"
                )
                self.rl_panel.set_training_active(False)
                return
            self.debug_panel.append_message(
                f"[RL] {algo} single-route training started "
                f"route={ep['label']}→{xp['label']} "
                f"episodes={cfg['max_episodes']} "
                f"max_steps={cfg['max_steps_per_episode']} seed={cfg['seed']} "
                f"speed={cfg['speed_config']['profile']}"
            )

        # 训练线程(独立 wire 实例,与 UI 隔离)
        try:
            from zmd.rl.trainer import PPOTrainer, SACTrainer, TD3Trainer, DDPGTrainer, DQNTrainer
        except ModuleNotFoundError as exc:
            self.debug_panel.append_message(
                f"[RL] {algo} training dependency missing: {exc.name}; "
                "install stable-baselines3 first"
            )
            self.rl_panel.set_training_active(False)
            self.rl_panel.set_status("Training dependency missing")
            return
        if algo == "PPO":
            trainer_cls = PPOTrainer
        elif algo == "SAC":
            trainer_cls = SACTrainer
        elif algo == "TD3":
            trainer_cls = TD3Trainer
        elif algo == "DDPG":
            trainer_cls = DDPGTrainer
        elif algo == "DQN":
            trainer_cls = DQNTrainer
        else:
            raise ValueError(f"Unsupported algo: {algo}")
        self._rl_thread = QThread(self)
        self._rl_trainer = trainer_cls(
            centerline=single_cl,
            centerlines=centerlines,
            sdf_fn=self._vessel_sdf(),
            reward_params=params,
            max_episodes=cfg["max_episodes"],
            max_steps_per_episode=cfg["max_steps_per_episode"],
            seed=cfg.get("seed", 0),
            speed_config=cfg.get("speed_config"),
        )
        self._rl_trainer.moveToThread(self._rl_thread)
        self._rl_thread.started.connect(self._rl_trainer.run)
        self._rl_trainer.episode_ended.connect(self._on_rl_episode_ended)
        self._rl_trainer.eval_done.connect(self._on_rl_eval_done)
        self._rl_trainer.status_changed.connect(self.rl_panel.set_status)
        self._rl_trainer.step_observed.connect(self._on_rl_step_observed)
        self._rl_trainer.finished.connect(self._on_rl_finished)
        self._rl_thread.start()

    def _on_rl_stop(self):
        self.debug_panel.append_message("[RL] Stop requested …")
        if getattr(self, "_rl_trainer", None) is not None:
            self._rl_trainer.request_stop()
        self.rl_panel.set_status("Stopping …")

    def _on_rl_step_observed(self, polyline, cam_pose):
        """训练 env 每 ~40ms 推一帧到这里 → 用 scene 0 的渲染器/inset 实时画。
        UI 的物理 wire 状态不变(只是 actor 被训练 wire 的 polyline 覆盖)。"""
        if polyline is None or len(polyline) < 2:
            return
        try:
            scene_index = 0
            renderer = self.scene_manager.renderers.get(scene_index, {}).get("guidewire")
            gw_radius = self.scene_manager.guidewire_radius.get(scene_index, 0.1)
            if renderer is not None:
                renderer.update_centerline(polyline, radius=gw_radius)
            inset = self.inset_views.get(scene_index)
            if inset is not None:
                inset.update_guidewire(polyline, gw_radius=gw_radius,
                                       camera_pose=cam_pose)
        except Exception:
            pass

    def _on_rl_episode_ended(self, payload: dict):
        """每个训练 episode 只更新状态文字(带探索噪声,不画曲线)。
        曲线由 _on_rl_eval_done 的周期性评估数据驱动(论文式)。"""
        ep_idx = int(payload.get("index", 0))
        ep_r = float(payload.get("reward", 0.0))
        avg_r = float(payload.get("avg_reward_50", 0.0))
        succ = float(payload.get("success_rate_50", 0.0))
        ep_len = int(payload.get("length", 0))
        ep_blocked = int(payload.get("blocked", 0))
        total = int(self.rl_panel.get_training_config()["max_episodes"])
        self.debug_panel.set_training_status(
            f"Training ep={ep_idx}/{total} r={ep_r:+.1f} avg50={avg_r:+.1f} "
            f"succ50={succ * 100:.0f}% len={ep_len} blk={ep_blocked}"
        )

    def _on_rl_eval_done(self, payload: dict):
        """周期性评估完成 → 画 4 条论文式评估曲线(X = 训练步数)。"""
        step = int(payload.get("timestep", 0))
        self.debug_panel.append_reward(0, float(payload.get("mean_reward", 0.0)))
        self.debug_panel.append_success_rate(0, float(payload.get("success_rate", 0.0)))
        self.debug_panel.append_episode_length(0, int(payload.get("mean_length", 0)))
        self.debug_panel.append_blocked_count(0, int(payload.get("mean_blocked", 0)))
        self.debug_panel.set_curves_x_axis_label(f"Training steps {step}")

    def _on_rl_finished(self, ckpt_path: str):
        # 训练线程退出 — 清理
        if getattr(self, "_rl_thread", None) is not None:
            self._rl_thread.quit()
            self._rl_thread.wait(3000)
            self._rl_thread = None
        self._rl_trainer = None
        self.rl_panel.set_training_active(False)
        # 训练结束清掉调试面板的状态行(下次开始训练才重显)
        self.debug_panel.set_training_status("")
        if ckpt_path:
            from pathlib import Path
            self.debug_panel.append_message(f"[RL] Saved {Path(ckpt_path).name}")
        else:
            self.debug_panel.append_message("[RL] Training terminated unexpectedly")

    def _build_route_centerline(self, ep: dict, xp: dict):
        """给定 (entry_preset, exit_preset) 构造一条裁剪好的 Centerline。失败返回 None。

        裁剪规则与 _rebuild_route 一致:entry 往内缩 DEFAULT_INWARD_OFFSET,
        exit 按 pullback_fraction 回退。
        """
        if not self.vtp_branches:
            return None
        try:
            branch = pick_route(self.vtp_branches,
                                np.asarray(ep["entry_point"]),
                                np.asarray(xp["exit_point"]))
            pts = np.asarray(branch.points)
            radii = branch.radius
            if np.linalg.norm(pts[0] - np.asarray(ep["entry_point"])) > \
               np.linalg.norm(pts[-1] - np.asarray(ep["entry_point"])):
                pts = pts[::-1].copy()
                if radii is not None:
                    radii = radii[::-1].copy()
            n = len(pts)
            eo = max(0, min(int(self.DEFAULT_INWARD_OFFSET), n - 4))
            pull_frac = float(xp.get("pullback_fraction",
                                     self.DEFAULT_EXIT_PULLBACK_FRACTION))
            xo = max(0, min(int(round(n * pull_frac)), n - eo - 2))
            tpts = pts[eo : n - xo]
            tr = radii[eo : n - xo] if radii is not None else None
            return Centerline(
                points=tpts, radius=tr,
                source_point=tpts[0].copy(),
                target_point=tpts[-1].copy(),
                source="vmtk",
                branches=self.vtp_branches,
            )
        except Exception as exc:
            self.debug_panel.append_message(
                f"[RL] Failed to build route {ep['label']}→{xp['label']}: {exc}"
            )
            return None

    def _build_all_route_centerlines(self):
        """枚举所有 (entry, exit) 组合,构造 Centerline 列表给泛化训练用。"""
        out = []
        for ep in DEFAULT_ENTRY_POINT_PRESETS:
            for xp in DEFAULT_EXIT_POINT_PRESETS:
                cl = self._build_route_centerline(ep, xp)
                if cl is not None:
                    out.append(cl)
        return out

    def _on_rl_load_model(self, path: str):
        """加载 ckpt 文件路径。实际 PPO.load 延迟到 _on_rl_run_model 时执行,
        避免没装 sb3 / 文件坏时启动就崩。"""
        self._rl_loaded_ckpt = str(path)
        self.debug_panel.append_message(f"[RL] Model selected: {path}")
        self.rl_panel.set_status(
            "Model path set; click ▶ Run Model to start inference"
        )

    def _on_rl_run_model(self):
        """用加载的 ckpt 驱动 UI 上的 active_guidewire 走完。"""
        if not self._rl_loaded_ckpt:
            self.debug_panel.append_message("[RL] Load a checkpoint first")
            self.rl_panel.set_inference_active(False)
            return
        scene_index = 0
        gw = self.active_guidewires.get(scene_index)
        if gw is None:
            self.debug_panel.append_message("[RL] No active guidewire in the current scene")
            self.rl_panel.set_inference_active(False)
            return
        # 训练态(若在跑)必须先停,否则两边都改 wire 状态
        if getattr(self, "_rl_trainer", None) is not None:
            self.debug_panel.append_message(
                "[RL] Stop training before running inference"
            )
            self.rl_panel.set_inference_active(False)
            return

        # 推理路线:如果用户取消"跟随场景 1",按 RL 面板下拉的 (entry, exit) 切换
        # 场景 1 的路线 → _rebuild_route 会重建 wire + centerline + 渲染
        route_cfg = self.rl_panel.get_inference_route()
        entry_idx = max(0, min(int(route_cfg["entry_idx"]),
                               len(DEFAULT_ENTRY_POINT_PRESETS) - 1))
        exit_idx = max(0, min(int(route_cfg["exit_idx"]),
                              len(DEFAULT_EXIT_POINT_PRESETS) - 1))
        if route_cfg["follow_scene"]:
            current_route = self.scene_state.get(scene_index) or {}
            current_entry = current_route.get("entry_preset")
            current_exit = current_route.get("exit_preset")
            entry_idx = next(
                (i for i, preset in enumerate(DEFAULT_ENTRY_POINT_PRESETS)
                 if preset is current_entry or preset == current_entry),
                entry_idx,
            )
            exit_idx = next(
                (i for i, preset in enumerate(DEFAULT_EXIT_POINT_PRESETS)
                 if preset is current_exit or preset == current_exit),
                exit_idx,
            )
        if not route_cfg["follow_scene"]:
            ep = DEFAULT_ENTRY_POINT_PRESETS[entry_idx]
            xp = DEFAULT_EXIT_POINT_PRESETS[exit_idx]
            cur = self.scene_state.get(scene_index) or {}
            if (cur.get("entry_preset") is not ep
                    or cur.get("exit_preset") is not xp):
                self.debug_panel.append_message(
                f"[RL] Inference route changed → {ep['label']} → {xp['label']}"
                )
                self.apply_entry_preset_route(ep)
                self.apply_exit_preset_route(xp)
                # _rebuild_route 重建了 wire,重新拿
                gw = self.active_guidewires.get(scene_index)
                if gw is None:
                    self.rl_panel.set_inference_active(False)
                    return

        from zmd.rl.inference import PPOInferenceController, SACInferenceController, TD3InferenceController, DDPGInferenceController, DQNInferenceController
        model_name = Path(self._rl_loaded_ckpt).name.lower()
        if model_name.startswith("sac_"):
            inference_cls = SACInferenceController
            model_algo = "SAC"
        elif model_name.startswith("ppo_"):
            inference_cls = PPOInferenceController
            model_algo = "PPO"
        elif model_name.startswith("td3_"):
            inference_cls = TD3InferenceController
            model_algo = "TD3"
        elif model_name.startswith("ddpg_"):
            inference_cls = DDPGInferenceController
            model_algo = "DDPG"
        elif model_name.startswith("dqn_"):
            inference_cls = DQNInferenceController
            model_algo = "DQN"
        else:
            from zmd.rl.inference_metrics import load_checkpoint_training_config
            saved_algo = str(
                load_checkpoint_training_config(
                    Path(self._rl_loaded_ckpt)
                ).get("algorithm", "")
            ).upper()
            model_algo = (
                saved_algo
                if saved_algo in {"PPO", "SAC", "TD3", "DDPG", "DQN"}
                else self.rl_panel.current_algorithm()
            )
            inference_cls = {
                "PPO": PPOInferenceController,
                "SAC": SACInferenceController,
                "TD3": TD3InferenceController,
                "DDPG": DDPGInferenceController,
                "DQN": DQNInferenceController,
            }.get(model_algo, PPOInferenceController)
        # if model_algo == "SAC":
        #     inference_cls = SACInferenceController
        #     model_algo = "SAC"
        #     elif model_name.startswith("ppo_"):
        #         inference_cls = PPOInferenceController
        #         model_algo = "PPO"
        #     elif model_name.startswith("td3_"):
        #         inference_cls = TD3InferenceController
        #         model_algo = "TD3"
        #     elif model_name.startswith("ddpg_"):
        #         inference_cls = DDPGInferenceController
        #         model_algo = "DDPG"
        #     inference_cls = SACInferenceController if model_algo == "SAC" else PPOInferenceController
        # 销毁旧 controller(每次跑都新建,避免状态残留)
        if self._rl_inference is not None:
            try: self._rl_inference.stop()
            except Exception: pass
        speed_config = self.rl_panel.get_speed_config()
        route_name = f"entry{entry_idx + 1}_exit{exit_idx + 1}"
        self._rl_inference = inference_cls(
            ckpt_path=self._rl_loaded_ckpt,
            guidewire=gw,
            parent=self,
            success_progress=self._rl_model_success_progress(self._rl_loaded_ckpt),
            speed_config=speed_config,
            output_dir=str(Path(self._rl_loaded_ckpt).parent / "inference_outputs"),
            run_metadata={
                "algorithm": model_algo,
                "entry": entry_idx + 1,
                "exit": exit_idx + 1,
                "route": route_name,
            },
        )
        self._rl_inference.status_changed.connect(self.rl_panel.set_status)
        self._rl_inference.finished.connect(self._on_rl_inference_finished)
        self._rl_inference.metrics_saved.connect(self._on_rl_inference_metrics_saved)
        if not self._rl_inference.load():
            self.rl_panel.set_inference_active(False)
            return
        self.debug_panel.append_message(
            f"[RL] {model_algo} inference replay started "
            f"ckpt={Path(self._rl_loaded_ckpt).name} "
            f"route={route_name} speed={speed_config['profile']}"
        )
        self._rl_inference.start()

    def _rl_model_success_progress(self, ckpt_path: str) -> float:
        """Return the inference success threshold, independent of training config."""
        return 0.99

    def _on_rl_stop_model(self):
        if self._rl_inference is not None:
            self._rl_inference.stop()
        self.rl_panel.set_inference_active(False)

    def _on_rl_inference_finished(self, success: bool):
        self.rl_panel.set_inference_active(False)
        msg = "destination reached" if success else "destination not reached"
        self.debug_panel.append_message(f"[RL] Inference finished: {msg}")

    def _on_rl_inference_metrics_saved(self, rollout_path: str, summary_path: str):
        self.debug_panel.append_message(
            f"[RL] Step-by-step inference metrics: {rollout_path}\n"
            f"[RL] Inference summary metrics: {summary_path}"
        )

    def _on_rl_reward_params_changed(self, params: dict):
        # 高频改动 — 只静默存到状态,不刷 log,避免刷屏。训练已经开始时,
        # 修改这里不会影响正在跑的训练(env 内部已经持有了 reward_params 副本)。
        self._rl_reward_params = dict(params)

    def _refresh_sidebar_buttons(self):
        expanded = self.sidebar.expanded
        mode = self.sidebar.current_mode()
        self.scene_edit_button.setText(
            "Hide Editor" if expanded and mode == "exp" else "Scene Editor"
        )
        self.guidewire_menu_button.setText(
            "Hide Guidewire" if expanded and mode == "guidewire" else "Guidewire"
        )

    def toggle_debug_panel(self):
        is_visible = not self.debug_panel_host.isVisible()
        self.debug_panel_host.setVisible(is_visible)
        self.debug_info_button.setText("Hide Debug" if is_visible else "Debug Info")
        self.update_debug_panel_width()

    # ------------------------------------------------------------------
    # 中心线加载
    # ------------------------------------------------------------------

    def _lazy_init_scene(self, scene_index):
        """延迟初始化指定场景 — 加载 mesh 后立即重建该场景的中心线路径。"""
        if self.scene_manager is None or scene_index in (None,):
            return
        if scene_index not in self.experiment_scenes:
            return
        try:
            self.scene_manager.ensure_initialized(scene_index)
        except Exception:
            return
        # 该场景的路径状态可能在 _load_vmtk_centerline 时已建好，但渲染未画
        st = self.scene_state.get(scene_index)
        if st and st.get("trimmed") is not None:
            self.scene_manager.load_centerline(st["trimmed"], scene_index=scene_index)
            trimmed_pts = st["trimmed"].points
            marker_radius = DEFAULT_VASCULAR_SETTINGS["marker_radius"]
            self.scene_manager.set_markers(scene_index, trimmed_pts[0], trimmed_pts[-1],
                                           marker_radius)
            self._update_progress_overlay(scene_index, 0)

    def _load_vmtk_centerline(self) -> None:
        """加载 VMTK 中心线树并为每个场景初始化默认路径。"""
        from zmd.utils.paths import ROOT_DIR

        vtp_path = ROOT_DIR / "projects" / "cache" / "VascuModel_centerline.vtp"
        branches = Centerline.load_branches_from_vtp(vtp_path)
        if not branches:
            self.vtp_branches = None
            self.debug_panel.append_message(
                f"VMTK centerline not found: {vtp_path.name}.\n"
                "Install VMTK with conda, then run: "
                "python scripts/vmtk_extract_centerline.py"
            )
            return
        self.vtp_branches = branches

        default_entry = DEFAULT_ENTRY_POINT_PRESETS[0]
        default_exit = DEFAULT_EXIT_POINT_PRESETS[2]
        for scene_index in self.experiment_scenes:
            self.scene_state[scene_index] = {
                "entry_preset": default_entry,
                "exit_preset": default_exit,
                "entry_offset": self.DEFAULT_INWARD_OFFSET,
                # None = 按 DEFAULT_EXIT_PULLBACK_FRACTION 在 _rebuild_route 解析
                "exit_offset": None,
                "full_branch": None,
                "trimmed": None,
            }
            self._rebuild_route(scene_index)

    def _init_simulation_engine(self) -> None:
        """已下沉到 _rebuild_route — 每场景独立一个 SimulationEngine。"""

    # ------------------------------------------------------------------
    # 路径裁剪 / 重建
    # ------------------------------------------------------------------

    def _rebuild_route(self, scene_index: int) -> None:
        """根据当前 (entry_preset, exit_preset, entry_offset, exit_offset) 重新生成路径，
        重新加载中心线、标记点、仿真后端。"""
        if not self.vtp_branches:
            return
        st = self.scene_state.get(scene_index)
        if not st: return
        ep = st["entry_preset"]
        xp = st["exit_preset"]
        branch = pick_route(self.vtp_branches,
                            np.asarray(ep["entry_point"]),
                            np.asarray(xp["exit_point"]))
        pts = np.asarray(branch.points)
        radii = branch.radius
        # 自动翻转使 start 离 source 更近
        if np.linalg.norm(pts[0] - np.asarray(ep["entry_point"])) > \
           np.linalg.norm(pts[-1] - np.asarray(ep["entry_point"])):
            pts = pts[::-1].copy()
            if radii is not None: radii = radii[::-1].copy()

        n = len(pts)
        eo = max(0, min(int(st["entry_offset"]), n - 4))
        # exit_offset 支持 None(按 branch 长度 × 出口预设的 pullback_fraction 算,
        # 没设就用全局 DEFAULT_EXIT_PULLBACK_FRACTION)。
        raw_xo = st["exit_offset"]
        if raw_xo is None:
            frac = float(
                st["exit_preset"].get(
                    "pullback_fraction", self.DEFAULT_EXIT_PULLBACK_FRACTION
                )
            )
            raw_xo = int(round(n * frac))
        xo = max(0, min(int(raw_xo), n - eo - 2))
        st["entry_offset"], st["exit_offset"] = eo, xo
        trimmed_pts = pts[eo : n - xo]
        trimmed_r = radii[eo : n - xo] if radii is not None else None

        cl = Centerline(
            points=trimmed_pts,
            radius=trimmed_r,
            source_point=trimmed_pts[0].copy(),
            target_point=trimmed_pts[-1].copy(),
            source="vmtk",
            branches=self.vtp_branches,
        )
        st["full_branch"] = branch
        st["trimmed"] = cl

        # 渲染中心线 + 标记
        self.scene_manager.load_centerline(cl, scene_index=scene_index)
        marker_radius = DEFAULT_VASCULAR_SETTINGS["marker_radius"]
        self.scene_manager.set_markers(scene_index, trimmed_pts[0], trimmed_pts[-1], marker_radius)
        # 默认导丝 — MuJoCo 医疗导丝
        guidewire = self._create_default_guidewire(scene_index, cl)
        if guidewire is not None:
            self.active_guidewires[scene_index] = guidewire
            engine = getattr(guidewire, "engine", None)
            if engine is not None:
                self.simulation_engines[scene_index] = engine
                initial_idx = engine.get_state().guidewire.inserted_index
            else:
                initial_idx = 0
        else:
            initial_idx = 0
        self.sidebar.set_guidewire_progress(scene_index, initial_idx, cl.n_points)
        self._update_progress_overlay(scene_index, initial_idx)
        self._update_minimap(scene_index, initial_idx)
        # 把场景 1 的当前路线提示同步到 RL 面板(在 "跟随场景 1" 模式下能看到走哪条)
        if scene_index == 0:
            try:
                ep_lbl = st.get("entry_preset", {}).get("label", "—")
                xp_lbl = st.get("exit_preset", {}).get("label", "—")
                self.rl_panel.set_scene_route_hint(f"{ep_lbl} → {xp_lbl}")
            except Exception:
                pass


    # 共享一份血管 SDF(vtkImplicitPolyDataDistance),全部 scene 用同一份血管 STL
    _vessel_sdf_fn = None

    def _vessel_sdf(self):
        """返回 fn(pos)->signed_distance,负 = lumen 内部,正 = 外部。

        被 MuJoCo 导丝创建时通过 set_collision_sdf 注入,做整段前端穿模检查。
        STL 由 ``projects/cache/VascuModel_collision.stl`` 提供,缺省回退到 project
        默认 vessel_model_path。VTK 计算开销大,所以做成类级缓存,4 个 scene 共享。
        """
        if type(self)._vessel_sdf_fn is not None:
            return type(self)._vessel_sdf_fn
        try:
            from pathlib import Path
            import pyvista as pv
            import vtk
            stl = Path("projects/cache/VascuModel_collision.stl").resolve()
            if not stl.exists():
                stl = Path(self.project_config.vessel_model_path)
            mesh = pv.read(str(stl))
            f = vtk.vtkImplicitPolyDataDistance()
            f.SetInput(mesh)
            def _sdf(p, _f=f):
                try:
                    px, py, pz = float(p[0]), float(p[1]), float(p[2])
                    return float(_f.EvaluateFunction([px, py, pz]))
                except Exception:
                    return 0.0
            type(self)._vessel_sdf_fn = _sdf
            # 自检
            _ = _sdf([0.0, 0.0, 0.0])
            print(f"[vessel sdf] loaded from {stl} ({mesh.n_cells} tris)", flush=True)
            return type(self)._vessel_sdf_fn
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[vessel sdf] Load failed: {exc!r}", flush=True)
            return None

    def _create_default_guidewire(self, scene_index, cl):
        """默认使用 MuJoCo 医疗导丝。无 MuJoCo 包则返回 None。"""
        if MujocoGuidewire is None:
            self.debug_panel.append_message(
                "[MuJoCo] Unavailable: the mujoco Python package is not installed "
                "(pip install mujoco)"
            )
            return None
        cb = lambda s, idx=scene_index: self._on_guidewire_state(idx, s)
        try:
            gw = MujocoGuidewire(cl, on_state_change=cb, config=self._mujoco_config)
        except Exception as exc:
            self.debug_panel.append_message(
                f"[MuJoCo] Initialization failed: {type(exc).__name__}: {exc}"
            )
            return None
        # 注入血管 SDF 做穿模硬安全检查(整段前端,不只 tip)
        try:
            sdf = self._vessel_sdf()
            if sdf is not None and hasattr(gw, "set_collision_sdf"):
                gw.set_collision_sdf(sdf)
                self.debug_panel.append_message(
                    f"[scene{scene_index+1}] MuJoCo + SDF collision checking enabled"
                )
            else:
                self.debug_panel.append_message(
                    f"[scene{scene_index+1}] SDF unavailable; "
                    "guidewire penetration checking skipped"
                )
        except Exception as exc:
            self.debug_panel.append_message(
                f"[scene{scene_index+1}] SDF injection failed: {exc}"
            )
        # 中心线辅助可视化默认隐藏(用户可在侧边栏「中心线」菜单打开)
        self._auto_hide_centerline(scene_index)
        return gw

    def _auto_hide_centerline(self, scene_index):
        """隐藏中心线辅助显示；用户可在「实验控制 → 中心线」菜单重新打开。"""
        try:
            cur = self.sidebar.centerline_states.get(scene_index)
            radius = float(cur.get("radius", 0.10)) if cur else 0.10
            show_all = bool(cur.get("show_all", True)) if cur else True
            self.scene_manager.apply_centerline_settings(scene_index, radius, False, show_all)
            self.sidebar.set_centerline_settings(scene_index, radius, False, show_all)
        except Exception:
            pass

    def _auto_show_centerline(self, scene_index):
        """切回中心线/几何导丝时恢复中心线辅助显示（保留 radius / show_all 现状）。"""
        try:
            cur = self.sidebar.centerline_states.get(scene_index)
            radius = float(cur.get("radius", 0.10)) if cur else 0.10
            show_all = bool(cur.get("show_all", True)) if cur else True
            self.scene_manager.apply_centerline_settings(scene_index, radius, True, show_all)
            self.sidebar.set_centerline_settings(scene_index, radius, True, show_all)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 同步辅助
    # ------------------------------------------------------------------

    def _target_indices(self, scene_index):
        if self.sidebar.sync_to_other_scenes_enabled():
            return list(self.experiment_scenes.keys())
        return [scene_index]

    def select_active_scene(self, scene_index):
        if scene_index not in self.experiment_scenes:
            return
        self.active_control_scene_index = scene_index
        self.sidebar.set_current_scene_index(scene_index)
        self.update_active_scene_highlight()

    def register_viewport_click_target(self, widget, scene_index):
        self.viewport_click_targets[widget] = scene_index
        widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        scene_index = self.viewport_click_targets.get(watched)
        if scene_index is not None and event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self.select_active_scene(scene_index)
        # scene / panel resize → 重新定位所有 overlay（progress + inset + minimap）
        if event.type() == QEvent.Type.Resize:
            for s_idx, s_widget in self.experiment_scenes.items():
                if watched is s_widget:
                    self._reposition_all_overlays(s_idx)
                    break
            for s_idx, p_widget in self.viewport_panels.items():
                if watched is p_widget:
                    self._reposition_all_overlays(s_idx)
                    break
        # 全局键盘：↑/↓ 控制导丝（spinbox / combobox / lineedit 有焦点时跳过）
        focus_widget = QApplication.focusWidget()
        skip_keys = isinstance(focus_widget, (QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit))
        if not skip_keys and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                if not event.isAutoRepeat():
                    self._key_step_delta = 3 if key == Qt.Key.Key_Up else -3
                    self._on_key_step_tick()
                    self._key_step_timer.start()
                event.accept(); return True
            if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                # 长按 ←/→ 连续旋转(模拟介入手柄持续转动)。
                # 原实现只在 KeyPress 非 auto-repeat 时触发一次,长按完全无响应。
                if not event.isAutoRepeat():
                    self._key_rotate_dir = 1 if key == Qt.Key.Key_Right else -1
                    self._on_key_rotate_tick()
                    self._key_rotate_timer.start()
                event.accept(); return True
        if not skip_keys and event.type() == QEvent.Type.KeyRelease:
            key = event.key()
            if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                if not event.isAutoRepeat():
                    self._key_step_timer.stop()
                    self._key_step_delta = 0
                event.accept(); return True
            if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                if not event.isAutoRepeat():
                    self._key_rotate_timer.stop()
                    self._key_rotate_dir = 0
                event.accept(); return True
        return super().eventFilter(watched, event)

    def _on_rotate_key(self, clockwise: bool):
        """← / → 键 — 仅对 supports_rotation=True 的导丝有效。
        中心线导丝静默，无 debug 噪音。"""
        scene_index = self._active_scene_for_keys()
        for target_index in self._target_indices(scene_index):
            gw = self.active_guidewires.get(target_index)
            if gw is None or not gw.supports_rotation():
                continue
            if clockwise:
                gw.rotate_clockwise()
            else:
                gw.rotate_counterclockwise()

    def make_viewport_tool_button(self, icon_name, tooltip, handler, checkable=False):
        button = QToolButton()
        button.setObjectName("ViewportToolButton")
        button.setIcon(self.viewport_icon(icon_name))
        button.setIconSize(QSize(16, 16))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setCheckable(checkable)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(handler)
        return button

    def viewport_icon(self, name):
        if name not in self.viewport_icon_cache:
            self.viewport_icon_cache[name] = QIcon(self.draw_viewport_icon(name))
        return self.viewport_icon_cache[name]

    def draw_viewport_icon(self, name):
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#dbe7f6"), 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if name == "camera_top":
            painter.drawEllipse(5, 5, 10, 10)
            painter.drawLine(10, 2, 10, 6)
            painter.drawLine(10, 14, 10, 18)
            painter.drawLine(2, 10, 6, 10)
            painter.drawLine(14, 10, 18, 10)
        elif name == "camera_side":
            painter.drawRect(4, 7, 12, 7)
            painter.drawLine(5, 7, 9, 4)
            painter.drawLine(15, 7, 11, 4)
            painter.drawLine(9, 4, 11, 4)
        elif name == "camera_front":
            painter.drawRect(4, 5, 12, 10)
            painter.drawEllipse(8, 8, 4, 4)
        elif name == "camera_reset":
            painter.drawArc(4, 4, 12, 12, 35 * 16, 285 * 16)
            painter.drawLine(15, 4, 16, 9)
            painter.drawLine(15, 4, 10, 4)
        elif name == "follow":
            painter.drawEllipse(5, 5, 10, 10)
            painter.drawLine(10, 3, 10, 7)
            painter.drawLine(10, 13, 10, 17)
            painter.drawLine(3, 10, 7, 10)
            painter.drawLine(13, 10, 17, 10)
        elif name == "minimap":
            painter.drawRect(3, 4, 14, 12)
            painter.drawLine(7, 4, 7, 16)
            painter.drawLine(13, 4, 13, 16)
            painter.drawLine(3, 9, 17, 9)
            painter.drawLine(3, 13, 17, 13)
        elif name == "zoom":
            painter.drawLine(4, 8, 4, 4)
            painter.drawLine(4, 4, 8, 4)
            painter.drawLine(12, 4, 16, 4)
            painter.drawLine(16, 4, 16, 8)
            painter.drawLine(16, 12, 16, 16)
            painter.drawLine(16, 16, 12, 16)
            painter.drawLine(8, 16, 4, 16)
            painter.drawLine(4, 16, 4, 12)
        elif name == "restore":
            painter.drawLine(8, 4, 8, 8)
            painter.drawLine(8, 8, 4, 8)
            painter.drawLine(12, 8, 16, 8)
            painter.drawLine(12, 8, 12, 4)
            painter.drawLine(12, 12, 12, 16)
            painter.drawLine(12, 12, 16, 12)
            painter.drawLine(8, 12, 4, 12)
            painter.drawLine(8, 12, 8, 16)
        elif name in {"eye", "eye_off"}:
            painter.drawEllipse(4, 7, 12, 6)
            painter.drawEllipse(8, 9, 4, 4)
            if name == "eye_off":
                painter.drawLine(4, 16, 16, 4)
        painter.end()
        return pixmap

    def update_active_scene_highlight(self):
        for scene_index, panel in self.viewport_panels.items():
            panel.setProperty("activeControl", scene_index == self.active_control_scene_index)
            panel.style().unpolish(panel)
            panel.style().polish(panel)

    def toggle_scene_zoom(self, scene_index):
        self.zoomed_scene_index = None if self.zoomed_scene_index == scene_index else scene_index
        self.apply_viewport_zoom_layout()

    def toggle_scene_mode(self):
        """单场景(只显示 scene 0) ↔ 多场景(2×2 全部 4 个)切换。"""
        self.single_scene_mode = not self.single_scene_mode
        if self.single_scene_mode:
            self.zoomed_scene_index = 0
            self.scene_mode_button.setText("Multi-Scene")
        else:
            self.zoomed_scene_index = None
            self.scene_mode_button.setText("Single Scene")
        self.apply_viewport_zoom_layout()
        self._apply_single_scene_ui()

    def _apply_single_scene_ui(self):
        """单场景模式:隐藏 viewport 的 zoom/hide 按钮(只有一个场景,缩小无意义)
        + 通知 debug 面板切到单场景标题/图例 + 通知 sidebar 收起 scene selector。"""
        for btn in self.zoom_buttons.values():
            btn.setVisible(not self.single_scene_mode)
        for btn in self.hide_buttons.values():
            btn.setVisible(not self.single_scene_mode)
        try:
            self.debug_panel.set_single_scene_mode(self.single_scene_mode)
        except Exception:
            pass
        try:
            self.sidebar.set_single_scene_mode(self.single_scene_mode)
        except Exception:
            pass

    def apply_viewport_zoom_layout(self):
        if not self.viewport_grid:
            return
        if self.zoomed_scene_index is None:
            visible_indices = set(self.viewport_panels)
            self.viewport_grid.setSpacing(self.viewport_grid_spacing)
            for row in range(2):
                self.viewport_grid.setRowStretch(row, 1)
            for column in range(2):
                self.viewport_grid.setColumnStretch(column, 1)
        else:
            visible_indices = {self.zoomed_scene_index}
            row, column = self.viewport_positions[self.zoomed_scene_index]
            self.viewport_grid.setSpacing(0)
            for row_index in range(2):
                self.viewport_grid.setRowStretch(row_index, 1 if row_index == row else 0)
            for column_index in range(2):
                self.viewport_grid.setColumnStretch(column_index, 1 if column_index == column else 0)
        for index, panel in self.viewport_panels.items():
            panel.setVisible(index in visible_indices)
            panel.updateGeometry()
        for index, button in self.zoom_buttons.items():
            is_zoomed = index == self.zoomed_scene_index
            button.setChecked(is_zoomed)
            button.setIcon(self.viewport_icon("restore" if is_zoomed else "zoom"))
            button.setToolTip(
                "Restore four views" if is_zoomed else "Maximize this view"
            )
        self.viewport_grid.invalidate()
        self.updateGeometry()
        QTimer.singleShot(0, self.render_visible_scenes)

    def render_visible_scenes(self):
        for index, scene in self.experiment_scenes.items():
            panel = self.viewport_panels.get(index)
            if panel and panel.isVisible():
                scene.plotter.render()

    def toggle_scene_visibility(self, scene_index):
        scene_widget = self.viewport_scene_widgets.get(scene_index)
        if not scene_widget:
            return
        scene_widget.setVisible(not scene_widget.isVisible())
        button = self.hide_buttons.get(scene_index)
        if button:
            is_hidden = not scene_widget.isVisible()
            button.setChecked(is_hidden)
            button.setIcon(self.viewport_icon("eye_off" if is_hidden else "eye"))
            button.setToolTip(
                "Show 3D viewport" if is_hidden else "Hide 3D viewport"
            )

    def update_debug_panel_width(self):
        if not self.debug_panel_host:
            return
        self.debug_panel_host.layout().setContentsMargins(0, 12, 0, 0)

    def handle_scene_status(self, scene_index, message):
        self.debug_panel.append_message(message, scene_index=scene_index)

    def apply_model_transform(self, scene_index, model_key, position, rotation, scale):
        target_indices = [scene_index]
        if self.sidebar.sync_to_other_scenes_enabled():
            target_indices = list(self.experiment_scenes.keys())
        actual_position, actual_rotation = self.to_actual_transform(model_key, position, rotation)
        for target_index in target_indices:
            scene = self.experiment_scenes.get(target_index)
            if scene:
                scene.set_model_transform(model_key, actual_position, actual_rotation, scale)
            if target_index != scene_index:
                self.sidebar.set_transform(target_index, model_key, position, rotation, scale)

    def apply_vascular_settings(self, scene_index, opacity, entry_point, exit_point, marker_radius):
        """仅处理透明度同步 — 入口/出口路径由 entry_preset_changed / exit_preset_changed 单独处理。"""
        del entry_point, exit_point, marker_radius
        for target_index in self._target_indices(scene_index):
            scene = self.experiment_scenes.get(target_index)
            if scene:
                scene.set_model_opacity("vascular", opacity)
            if target_index != scene_index:
                self.sidebar.set_vascular_settings(target_index, opacity, (0, 0, 0), (0, 0, 0), 0.3)

    DEFAULT_INWARD_OFFSET = 2  # 切换预设时 entry seed 默认往内部缩这么多步
    DEFAULT_EXIT_PULLBACK_FRACTION = 0.10
    """出口默认沿中心线回退分支长度的此比例(占 branch 点数,~10%)。

    每段分支长度不同,固定 N 步在短分支太多、在长分支太少 —— 改用分支长度的
    比例,3 个出口都自动适配。在 _rebuild_route 里把 exit_offset=None 解析为
    int(n_points * fraction)。"""

    def apply_entry_preset_route(self, preset):
        scene_index = self.sidebar.current_scene_index()
        for target_index in self._target_indices(scene_index):
            st = self.scene_state.get(target_index)
            if st is None: continue
            st["entry_preset"] = preset
            st["entry_offset"] = self.DEFAULT_INWARD_OFFSET
            self._rebuild_route(target_index)
        # 同步选择器 - 机器人位置
        robot_pos = tuple(preset.get("robot_position", (0.0, 0.0, 0.0)))
        self.apply_entry_preset_robot_position(scene_index, robot_pos)

    def apply_exit_preset_route(self, preset):
        scene_index = self.sidebar.current_scene_index()
        for target_index in self._target_indices(scene_index):
            st = self.scene_state.get(target_index)
            if st is None: continue
            st["exit_preset"] = preset
            # None = 让 _rebuild_route 按 branch 长度 ×10% 回退(每个出口分支
            # 长度不同,固定 step 数会让短分支退太多/长分支退太少)
            st["exit_offset"] = None
            self._rebuild_route(target_index)

    def step_entry_seed(self, scene_index, delta):
        """入口 seed 沿中心线滑动 delta 个采样点（>0 进入血管内部）。"""
        for target_index in self._target_indices(scene_index):
            st = self.scene_state.get(target_index)
            if st is None: continue
            st["entry_offset"] = max(0, st["entry_offset"] + int(delta))
            self._rebuild_route(target_index)

    def step_exit_seed(self, scene_index, delta):
        for target_index in self._target_indices(scene_index):
            st = self.scene_state.get(target_index)
            if st is None: continue
            st["exit_offset"] = max(0, st["exit_offset"] + int(delta))
            self._rebuild_route(target_index)

    def apply_centerline_settings(self, scene_index, radius, visible, show_all):
        for target_index in self._target_indices(scene_index):
            self.scene_manager.apply_centerline_settings(target_index, radius, visible, show_all)
            if target_index != scene_index:
                self.sidebar.set_centerline_settings(target_index, radius, visible, show_all)

    def apply_guidewire_radius(self, scene_index, radius):
        for target_index in self._target_indices(scene_index):
            self.scene_manager.apply_guidewire_radius(target_index, radius)
            if target_index != scene_index:
                self.sidebar.set_guidewire_settings(target_index, radius)

    def step_guidewire(self, scene_index, delta):
        """单步推进/后退当前选中的导丝。走 active_guidewire.step(action) 统一入口。"""
        action = GuidewireAction.ADVANCE if delta > 0 else GuidewireAction.RETRACT
        for target_index in self._target_indices(scene_index):
            gw = self.active_guidewires.get(target_index)
            if gw is None: continue
            gw.step(action)
            # 状态变化通过 on_state_change 回调 → _on_guidewire_state

    # MuJoCo 导丝调试输出节流：每场景独立计数，每 N 步输出一次
    _mujoco_debug_step: dict = {}       # scene_idx -> int（调用计数）
    _MUJOCO_DEBUG_INTERVAL = 15         # 每 15 步输出一次常规统计

    def _on_guidewire_state(self, scene_index, state):
        """active_guidewire 推进后回调 — 刷新所有 UI。未插入 (centerline 空) 时隐藏 actor。"""
        if state is None or state.guidewire is None:
            return
        gw_state = state.guidewire
        polyline = gw_state.centerline if gw_state.centerline is not None else None
        inserted = polyline is not None and len(polyline) >= 2

        renderer = self.scene_manager.renderers.get(scene_index, {}).get("guidewire")
        if not inserted:
            # 未插入 — 隐藏 actor
            if renderer is not None:
                renderer.set_visible(False)
        else:
            if renderer is not None and not renderer.visible:
                renderer.set_visible(True)
            self.scene_manager.update_from_state(state, scene_index=scene_index)
        total = self.scene_state[scene_index]["trimmed"].n_points
        self.sidebar.set_guidewire_progress(scene_index, gw_state.inserted_index, total)
        inset = self.inset_views.get(scene_index)
        if inset is not None and inset.isVisible():
            gw_radius = self.scene_manager.guidewire_radius.get(
                scene_index, DEFAULT_GUIDEWIRE_SETTINGS["radius"])
            if inserted:
                cam_pose = self._inset_camera_pose_for(scene_index)
                inset.update_guidewire(polyline, gw_radius=gw_radius,
                                       camera_pose=cam_pose)
            else:
                inset.clear_guidewire()
        self._update_minimap(scene_index, gw_state.inserted_index)

        # ---- MuJoCo 导丝调试信息推送到调试框 ----
        try:
            gw = self.active_guidewires.get(scene_index)
            if MujocoGuidewire is not None and isinstance(gw, MujocoGuidewire):
                self._push_mujoco_debug(scene_index, gw)
        except Exception:
            pass

    def _inset_camera_pose_for(self, scene_index: int):
        """当前场景导丝的内窥镜小窗相机位姿(仅 MuJoCo 导丝提供腔内安全相机)。

        其它导丝类型返回 None,inset 退回自带的 tip 切向相机逻辑。
        """
        try:
            gw = self.active_guidewires.get(scene_index)
            if MujocoGuidewire is not None and isinstance(gw, MujocoGuidewire):
                return gw.get_inset_camera_pose()
        except Exception:
            pass
        return None

    def _push_mujoco_debug(self, scene_index: int, gw) -> None:
        """把 MuJoCo 导丝 last_constraint_stats 推送到调试框（有节流）。

        - wall_slide / sdf_blocked 事件立即输出（不受节流限制）。
        - ok 状态每 _MUJOCO_DEBUG_INTERVAL 步输出一次常规统计。
        """
        try:
            s = gw.backend.last_constraint_stats
            status = s.get("status", "ok")
            cnt = self._mujoco_debug_step.get(scene_index, 0) + 1
            self._mujoco_debug_step[scene_index] = cnt

            # 立即上报：碰壁事件
            if status in ("wall_slide", "sdf_blocked"):
                ws_ok = s.get("wall_slide_success", False)
                reason = s.get("wall_slide_reason", "")
                min_sdf = s.get("min_sdf", float("nan"))
                d_cl = s.get("tip_to_centerline_distance", 0.0)
                self.debug_panel.append_message(
                    f"[MuJoCo wall contact] status={status} | "
                    f"slide_ok={ws_ok} | reason={reason} | "
                    f"min_sdf={min_sdf:.3f} | d_cl={d_cl:.2f}mm",
                    scene_index=scene_index,
                )
                return  # 碰壁帧不再输出常规统计

            # 常规统计（节流）
            if cnt % self._MUJOCO_DEBUG_INTERVAL == 0:
                ins = s.get("inserted_length", 0.0)
                trav = s.get("travel_length", 0.0)
                prog = s.get("progress", 0.0)
                d_cl = s.get("tip_to_centerline_distance", 0.0)
                sdf_en = s.get("sdf_enabled", False)
                # J-tip 转向信息(角度>0 时显示 twist 与有效权重)
                jtip = ""
                if s.get("j_tip_weight_effective", 0.0) > 0.0 or \
                        s.get("j_tip_angle_deg", 0.0) > 0.0:
                    jtip = (f" | Jtip={s.get('j_tip_angle_deg',0):.0f}° "
                            f"twist={s.get('twist_angle',0):.2f} "
                            f"w={s.get('j_tip_weight_effective',0):.2f}")
                self.debug_panel.append_message(
                    f"[MuJoCo] ins={ins:.1f}/{trav:.1f}mm prog={prog*100:.1f}% "
                    f"d_cl={d_cl:.2f}mm sdf={sdf_en}{jtip}",
                    scene_index=scene_index,
                )
        except Exception:
            pass

    # ----- 抽象：active guidewire API（未来物理导丝可替换实现） -----
    def _active_guidewire_advance(self, scene_index, delta):
        """当前选中导丝的前进 / 后退入口。currently only centerline guidewire."""
        self.step_guidewire(scene_index, delta)

    def _active_guidewire_reset(self, scene_index):
        self.reset_guidewire(scene_index)

    def _update_minimap(self, scene_index, inserted_index):
        mm = self.minimaps.get(scene_index)
        st = self.scene_state.get(scene_index)
        if mm is None or st is None or st.get("trimmed") is None:
            return
        pts = np.asarray(st["trimmed"].points)
        if 0 <= inserted_index < len(pts) - 1:
            tail = pts[inserted_index:]
            d = float(np.sum(np.linalg.norm(np.diff(tail, axis=0), axis=1)))
        else:
            d = 0.0
        mm.set_path(pts)
        mm.set_progress(int(inserted_index), d)

    def _update_progress_overlay(self, scene_index, inserted_index):
        """已废弃 — 进度信息现在由 minimap 显示。保留 no-op 以兼容调用方。"""
        del scene_index, inserted_index

    def _on_rotate_cw_button(self, scene_index):
        for target_index in self._target_indices(scene_index):
            gw = self.active_guidewires.get(target_index)
            if gw is not None and gw.supports_rotation():
                gw.rotate_clockwise()

    def _on_rotate_ccw_button(self, scene_index):
        for target_index in self._target_indices(scene_index):
            gw = self.active_guidewires.get(target_index)
            if gw is not None and gw.supports_rotation():
                gw.rotate_counterclockwise()

    def _on_hold_forward(self, scene_index):
        for target_index in self._target_indices(scene_index):
            gw = self.active_guidewires.get(target_index)
            if gw is not None: gw.start_hold_forward()

    def _on_hold_backward(self, scene_index):
        for target_index in self._target_indices(scene_index):
            gw = self.active_guidewires.get(target_index)
            if gw is not None: gw.start_hold_backward()

    def _on_hold_released(self, scene_index):
        for target_index in self._target_indices(scene_index):
            gw = self.active_guidewires.get(target_index)
            if gw is not None: gw.stop_hold()

    def _on_guidewire_color(self, scene_index, color):
        for target_index in self._target_indices(scene_index):
            gw = self.active_guidewires.get(target_index)
            if gw is not None: gw.set_color(color)
            self.scene_manager.apply_guidewire_color(target_index, color)
            inset = self.inset_views.get(target_index)
            if inset is not None and hasattr(inset, "set_color"):
                inset.set_color(color)
        self.debug_panel.append_message(f"Guidewire color updated → {color}")

    def reset_guidewire(self, scene_index):
        for target_index in self._target_indices(scene_index):
            gw = self.active_guidewires.get(target_index)
            if gw is None: continue
            gw.reset()
            # 状态会通过 on_state_change 回调 _on_guidewire_state 自动刷新

    # ------------------------------------------------------------------
    # 键盘 ↑/↓ 控制导丝推进/后退
    # ------------------------------------------------------------------

    def _active_scene_for_keys(self):
        """优先用当前侧边栏选择的场景，否则用 0。"""
        idx = self.sidebar.current_scene_index()
        return idx if idx is not None else 0

    def _on_key_step_tick(self):
        if self._key_step_delta:
            self._active_guidewire_advance(self._active_scene_for_keys(),
                                           self._key_step_delta)

    def _on_key_rotate_tick(self):
        if self._key_rotate_dir == 0:
            return
        self._on_rotate_key(self._key_rotate_dir > 0)

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key.Key_Up:
            self._key_step_delta = 3
            self._on_key_step_tick()
            self._key_step_timer.start()
            event.accept(); return
        if event.key() == Qt.Key.Key_Down:
            self._key_step_delta = -3
            self._on_key_step_tick()
            self._key_step_timer.start()
            event.accept(); return
        # Z → toggle 血管 STL 线框 / 实体表面
        if event.key() == Qt.Key.Key_Z:
            self._toggle_ground_grid_current_scene()
            event.accept(); return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            self._key_step_timer.stop()
            self._key_step_delta = 0
            event.accept(); return
        super().keyReleaseEvent(event)

    def _on_scene_key_forwarded(self, scene_index: int, key: int, pressed: bool) -> None:
        """3D 视窗转发过来的 Z — 当前激活场景才响应。"""
        qkey = Qt.Key(key)
        if qkey == Qt.Key.Key_Z:
            if pressed:
                for idx in self._target_indices(scene_index):
                    try:
                        self.scene_manager.toggle_vessel_wireframe(idx)
                    except Exception:
                        pass
            return

    def _toggle_ground_grid_current_scene(self) -> None:
        """Z 键 — workspace 层 fallback：当 3D 视窗没有焦点时也能响应。"""
        scene_index = self._active_scene_for_keys()
        for target_index in self._target_indices(scene_index):
            try:
                self.scene_manager.toggle_vessel_wireframe(target_index)
            except Exception:
                pass

    def toggle_minimap(self, scene_index, enabled):
        """minimap 开关 — sync ON 时广播到全部场景。"""
        for target_index in self._target_indices(scene_index):
            mm = self.minimaps.get(target_index)
            if mm is None: continue
            if enabled:
                mm.show(); mm.raise_(); mm.update()
            else:
                mm.hide()
            btn = self.minimap_buttons.get(target_index)
            if btn is not None and target_index != scene_index:
                btn.blockSignals(True)
                btn.setChecked(enabled)
                btn.blockSignals(False)

    def apply_camera_preset(self, scene_index, preset):
        """sync ON 时把相机预设广播到全部场景；sync OFF 时只动当前场景。"""
        for target_index in self._target_indices(scene_index):
            scene = self.experiment_scenes.get(target_index)
            if scene:
                scene.set_camera_preset(preset)

    def toggle_follow(self, scene_index, enabled):
        """follow 不再改主相机 — 控制右上角 inset 显隐。同步规则广播到全部目标场景。"""
        for target_index in self._target_indices(scene_index):
            self.scene_manager.set_follow_enabled(target_index, enabled)
            btn = self.follow_buttons.get(target_index)
            if btn is not None and target_index != scene_index:
                btn.blockSignals(True)
                btn.setChecked(enabled)
                btn.blockSignals(False)
            if btn is not None:
                btn.setToolTip(
                    "Close guidewire inset" if enabled else "Open guidewire inset"
                )
            inset = self.inset_views.get(target_index)
            if inset is None: continue
            if enabled:
                self._ensure_inset_vessel_loaded(target_index)
                inset.show()
                inset.raise_()
                # 立刻按当前 active_guidewire 状态绘制；未插入时 inset 自己显示占位
                gw = self.active_guidewires.get(target_index)
                if gw is not None:
                    state = gw.get_intervention_state()
                    if state is not None and state.guidewire is not None:
                        polyline = state.guidewire.centerline
                        gw_radius = getattr(gw, "radius",
                            self.scene_manager.guidewire_radius.get(
                                target_index, DEFAULT_GUIDEWIRE_SETTINGS["radius"]))
                        if polyline is not None and len(polyline) >= 2:
                            cam_pose = self._inset_camera_pose_for(target_index)
                            inset.update_guidewire(polyline, gw_radius=gw_radius,
                                                   camera_pose=cam_pose)
                        else:
                            inset.clear_guidewire()
            else:
                inset.hide()

    def _ensure_inset_vessel_loaded(self, scene_index):
        inset = self.inset_views.get(scene_index)
        if inset is None or inset.vessel_actor is not None:
            return
        path = self.project_config.vessel_model_path
        if not path or not path.exists():
            return
        try:
            mesh = self.mesh_cache.get_normalized(path,
                normalized_length=900.0)
            inset.load_vessel(mesh)
        except Exception:
            pass

    def apply_entry_preset_robot_position(self, scene_index, robot_position):
        target_indices = [scene_index]
        if self.sidebar.sync_to_other_scenes_enabled():
            target_indices = list(self.experiment_scenes.keys())
        robot_rotation = (0.0, 0.0, 0.0)
        actual_position, actual_rotation = self.to_actual_transform("robot", robot_position, robot_rotation)
        for target_index in target_indices:
            scene = self.experiment_scenes.get(target_index)
            if scene:
                scene.set_model_transform("robot", actual_position, actual_rotation, 0.1)
            self.sidebar.set_transform(target_index, "robot", robot_position, robot_rotation, 0.1)

    def sync_current_scene_to_others(self, scene_index):
        # 机器人 transform
        if scene_index in self.sidebar.robot_transforms:
            pos, rot = self.sidebar.robot_transforms[scene_index]
            self.apply_model_transform(scene_index, "robot", pos, rot, 0.1)
        # 血管透明度
        v_state = self.sidebar.vascular_states.get(scene_index)
        if v_state:
            self.apply_vascular_settings(
                scene_index, v_state.get("opacity", 0.5), (0, 0, 0), (0, 0, 0), 0.3
            )
        # 中心线参数
        c_state = self.sidebar.centerline_states.get(scene_index)
        if c_state:
            self.apply_centerline_settings(
                scene_index, c_state["radius"], c_state["visible"], c_state["show_all"]
            )
        # 导丝参数
        g_state = self.sidebar.guidewire_states.get(scene_index)
        if g_state:
            self.apply_guidewire_radius(scene_index, g_state["radius"])
        # 路径
        st = self.scene_state.get(scene_index)
        if st:
            for target_index in self.experiment_scenes:
                if target_index == scene_index: continue
                tst = self.scene_state[target_index]
                tst["entry_preset"] = st["entry_preset"]
                tst["exit_preset"] = st["exit_preset"]
                tst["entry_offset"] = st["entry_offset"]
                tst["exit_offset"] = st["exit_offset"]
                self._rebuild_route(target_index)

    def to_actual_transform(self, model_key, relative_position, relative_rotation):
        base_transform = DEFAULT_MODEL_TRANSFORMS.get(model_key, DEFAULT_MODEL_TRANSFORMS["vascular"])
        actual_position = tuple(b + r for b, r in zip(base_transform["position"], relative_position))
        actual_rotation = tuple(b + r for b, r in zip(base_transform["rotation"], relative_rotation))
        return actual_position, actual_rotation

    def build_viewport_grid(self):
        grid = QGridLayout()
        self.viewport_grid = grid
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(self.viewport_grid_spacing)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout_key = self.project_config.scene_layout
        experiments = SCENE_LAYOUTS.get(layout_key, SCENE_LAYOUTS["four_view"])
        for title, scene_index, row, column in experiments:
            self.viewport_positions[scene_index] = (row, column)
            grid.addWidget(self.build_experiment_panel(title, scene_index), row, column)
        return grid

    def build_experiment_panel(self, title, scene_index):
        panel = QFrame()
        panel.setObjectName("ViewportPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setObjectName("ViewportTitle")
        title_row.addWidget(title_label)
        title_row.addStretch(1)

        scene = Scene3DWidget(camera_preset="isometric")
        # Z 从 3D 视窗转发到 workspace(toggle 血管线框/实体)
        scene.key_forwarded.connect(
            lambda key, pressed, idx=scene_index: self._on_scene_key_forwarded(idx, key, pressed))

        top_button = self.make_viewport_tool_button("camera_top", "Top View", lambda _, idx=scene_index: self.apply_camera_preset(idx, "top"))
        side_button = self.make_viewport_tool_button("camera_side", "Side View", lambda _, idx=scene_index: self.apply_camera_preset(idx, "left"))
        front_button = self.make_viewport_tool_button("camera_front", "Front View", lambda _, idx=scene_index: self.apply_camera_preset(idx, "front"))
        reset_button = self.make_viewport_tool_button("camera_reset", "Reset Camera", lambda _, idx=scene_index: self.apply_camera_preset(idx, "isometric"))
        follow_button = self.make_viewport_tool_button(
            "follow", "Toggle guidewire inset",
            lambda checked, index=scene_index: self.toggle_follow(index, checked),
            checkable=True,
        )
        minimap_button = self.make_viewport_tool_button(
            "minimap", "Toggle bottom-left minimap",
            lambda checked, index=scene_index: self.toggle_minimap(index, checked),
            checkable=True,
        )
        minimap_button.setChecked(True)  # 默认显示
        zoom_button = self.make_viewport_tool_button("zoom", "Maximize this view", lambda _, index=scene_index: self.toggle_scene_zoom(index), checkable=True)
        hide_button = self.make_viewport_tool_button("eye", "Hide 3D viewport", lambda _, index=scene_index: self.toggle_scene_visibility(index), checkable=True)
        for button in (top_button, side_button, front_button, reset_button, follow_button, minimap_button, zoom_button, hide_button):
            title_row.addWidget(button)
        self.follow_buttons[scene_index] = follow_button
        self.minimap_buttons[scene_index] = minimap_button

        scene.status_changed.connect(lambda message, index=scene_index: self.handle_scene_status(index, message))

        # 模型加载和标记点设置已委托给 SceneManager
        self.experiment_scenes[scene_index] = scene
        self.viewport_panels[scene_index] = panel
        self.viewport_scene_widgets[scene_index] = scene
        self.zoom_buttons[scene_index] = zoom_button
        self.hide_buttons[scene_index] = hide_button
        panel.setProperty("activeControl", False)
        self.sidebar.set_transform(scene_index, "robot", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.1)
        self.sidebar.set_vascular_settings(
            scene_index,
            DEFAULT_VASCULAR_SETTINGS["opacity"],
            DEFAULT_VASCULAR_SETTINGS["entry_point"],
            DEFAULT_VASCULAR_SETTINGS["exit_point"],
            DEFAULT_VASCULAR_SETTINGS["marker_radius"],
        )
        self.sidebar.set_centerline_settings(
            scene_index,
            DEFAULT_CENTERLINE_SETTINGS["radius"],
            DEFAULT_CENTERLINE_SETTINGS["visible"],
            DEFAULT_CENTERLINE_SETTINGS["show_all_branches"],
        )
        self.sidebar.set_guidewire_settings(
            scene_index, DEFAULT_GUIDEWIRE_SETTINGS["radius"],
        )
        self.register_viewport_click_target(panel, scene_index)
        self.register_viewport_click_target(scene, scene_index)
        self.register_viewport_click_target(scene.plotter.interactor, scene_index)

        layout.addLayout(title_row)
        layout.addWidget(scene, 1)

        # 进度信息已下沉到 minimap，移除原本右下角的独立 progress label
        # 右上角导丝观察小窗 — 父对象为 panel（QFrame，非 native）
        inset = GuidewireInsetView(panel)
        inset.hide()
        self.inset_views[scene_index] = inset

        # 右下角 minimap — 父对象为 panel
        minimap = CenterlineMinimap(panel)
        minimap.show()
        minimap.raise_()
        self.minimaps[scene_index] = minimap

        scene.installEventFilter(self)        # 监听 scene resize 用同一 eventFilter
        panel.installEventFilter(self)
        QTimer.singleShot(50, lambda idx=scene_index: self._reposition_all_overlays(idx))
        QTimer.singleShot(500, lambda idx=scene_index: self._reposition_all_overlays(idx))
        return panel

    def _reposition_overlay(self, label, scene):
        label.adjustSize()
        label.move(scene.width() - label.width() - 14, scene.height() - label.height() - 14)

    def _reposition_all_overlays(self, scene_index):
        scene = self.experiment_scenes.get(scene_index)
        panel = self.viewport_panels.get(scene_index)
        if scene is None or panel is None:
            return
        # 两个 overlay 都贴右边，按 panel 尺寸自适应缩放
        OVERLAY_RIGHT = 16
        OVERLAY_TOP = 56
        OVERLAY_BOTTOM = 16
        OVERLAY_GAP = 12
        pw = max(panel.width(), 100)
        ph = max(panel.height(), 100)
        # 宽度 = panel 宽度的 28%，限制在 [200, 320]；高度按 4:3 / 比例算
        target_w = int(min(320, max(200, pw * 0.28)))
        inset_h = int(target_w * 0.72)
        mm_h = int(target_w * 0.78)
        # 二者总高 + gap 不超过 panel 高度 80%
        total_h = inset_h + mm_h + OVERLAY_GAP + OVERLAY_TOP + OVERLAY_BOTTOM
        max_total = int(ph * 0.92)
        if total_h > max_total:
            shrink = max_total / total_h
            target_w = int(target_w * shrink)
            inset_h = int(inset_h * shrink)
            mm_h = int(mm_h * shrink)

        inset = self.inset_views.get(scene_index)
        if inset is not None:
            inset.setFixedSize(target_w, inset_h)
            inset.move(pw - target_w - OVERLAY_RIGHT, OVERLAY_TOP)
            inset.raise_()
        mm = self.minimaps.get(scene_index)
        if mm is not None:
            mm.setFixedSize(target_w, mm_h)
            mm.move(pw - target_w - OVERLAY_RIGHT, ph - mm_h - OVERLAY_BOTTOM)
            mm.raise_()
            mm.update()
