"""强化学习面板 — 训练算法选择 / 开始 / 停止 / 加载模型 / 奖励参数调节。

当前阶段:
- UI 全部可见可点,但具体训练后端尚未接入(按钮信号已 emit,workspace 接的是 stub)
- 单场景训练版;未来多场景对比版会新开 UI

后端接入点(future):
- start_clicked / stop_clicked / load_model_requested(path: str) / reward_params_changed(dict)
- algorithm_changed(name: str)
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from zmd.ui.panels.control_sidebar import (
    FocusWheelDoubleSpinBox, make_section_label, make_step_button,
)
from zmd.rl.speed_schedule import SPEED_PROFILE_PRESETS, resolve_speed_config


RL_PANEL_STYLE = """
#RLPanel {
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(96, 165, 250, 0.55);
    border-radius: 14px;
}
#RLPanel QScrollArea, #RLScrollContent {
    background: transparent;
    border: none;
}
#RLPanelTitle {
    color: #dbeafe;
    font-size: 14px;
    font-weight: 900;
}
#RLPanelHint {
    color: #94a3b8;
    font-size: 11px;
}
#RLStatusLabel {
    color: #cbd5e1;
    font-size: 11px;
    padding: 4px 8px;
    background: rgba(2, 6, 23, 0.45);
    border: 1px solid rgba(96, 165, 250, 0.35);
    border-radius: 8px;
}
#RLStartButton {
    background: rgba(34, 197, 94, 0.78);
    border: 1px solid rgba(134, 239, 172, 0.9);
    border-radius: 9px;
    color: #ffffff;
    font-size: 12px;
    font-weight: 800;
    padding: 6px 10px;
}
#RLStartButton:hover { background: rgba(34, 197, 94, 0.95); }
#RLStartButton:disabled {
    background: rgba(71, 85, 105, 0.55);
    border-color: rgba(100, 116, 139, 0.6);
    color: #94a3b8;
}
#RLStopButton {
    background: rgba(239, 68, 68, 0.78);
    border: 1px solid rgba(252, 165, 165, 0.9);
    border-radius: 9px;
    color: #ffffff;
    font-size: 12px;
    font-weight: 800;
    padding: 6px 10px;
}
#RLStopButton:hover { background: rgba(239, 68, 68, 0.95); }
#RLStopButton:disabled {
    background: rgba(71, 85, 105, 0.55);
    border-color: rgba(100, 116, 139, 0.6);
    color: #94a3b8;
}
#RLToggleSection {
    color: #cbd5e1;
    font-size: 12px;
    font-weight: 700;
    padding: 4px 6px;
    border: none;
    text-align: left;
}
#RLToggleSection:hover { color: #f1f5f9; }
"""


# 默认奖励参数 — 来源于推荐配方(进度 100 × Δprog + 软 shaping)
DEFAULT_REWARD_PARAMS: dict = {
    "progress_weight":   200.0,   # 仅首次刷新最大进度时奖励
    "backtrack_weight":  100.0,   # progress 回退的惩罚权重
    "step_cost":         0.02,    # 每步小负
    "blocked_penalty":   0.5,     # status=blocked 每步扣
    "retract_cost":      0.1,    # 每次 retract 动作的额外代价
    "rotate_cost":       0.05,    # 每次 rotate 动作的额外代价
    "yaw_soft_thresh":   30.0,    # angle_to_centerline 超过此值开始软扣(度)
    "yaw_soft_weight":   0.01,    # 软扣系数(每度)
    "safety_margin_mm":  0.5,     # SDF 余量 < 此值开始扣
    "safety_weight":     0.5,     # 安全扣系数
    "success_bonus":     300.0,   # progress > 0.92 终局
    "timeout_penalty":   50.0,    # 超步惩罚
    "success_progress":  0.92,    # 训练和推理共用的到达终点阈值
}

# 支持的算法 — 与典型 stable-baselines3 / cleanrl 一致
RL_ALGORITHMS = ["PPO", "SAC", "TD3", "DDPG", "DQN"]


class RLPanel(QFrame):
    """折叠式强化学习面板。顶栏按钮 toggle 显隐;默认折叠。

    Signals(暴露给 workspace 接后端):
      algorithm_changed(str)             — 用户选了别的算法
      start_clicked()                    — 点开始训练
      stop_clicked()                     — 点停止训练
      load_model_requested(str)          — 用户选了 ckpt 文件
      reward_params_changed(dict)        — 任一奖励参数改动
    """

    EXPANDED_WIDTH = 380

    toggled = pyqtSignal(bool)
    algorithm_changed = pyqtSignal(str)
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    load_model_requested = pyqtSignal(str)
    reward_params_changed = pyqtSignal(dict)
    # 推理回放
    run_model_clicked = pyqtSignal()       # ▶ 运行模型
    stop_model_clicked = pyqtSignal()      # ■ 停止推理

    def __init__(self):
        super().__init__()
        self.setObjectName("RLPanel")
        self.setStyleSheet(RL_PANEL_STYLE)
        self.expanded = False
        self.setFixedWidth(0)
        self.setVisible(False)
        self._suspend = False
        self._reward_spins: dict = {}     # key → FocusWheelDoubleSpinBox
        self._reward_visible = False
        self._build_ui()

    # ------------------------------------------------------------------
    # 折叠/展开
    # ------------------------------------------------------------------

    def toggle(self):
        self.expanded = not self.expanded
        self.setVisible(self.expanded)
        self.setFixedWidth(self.EXPANDED_WIDTH if self.expanded else 0)
        self.toggled.emit(self.expanded)

    # ------------------------------------------------------------------
    # UI 构造
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setObjectName("RLScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        inner = QWidget()
        inner.setObjectName("RLScrollContent")
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        layout = QVBoxLayout(inner)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # 标题
        title = QLabel("Reinforcement Learning")
        title.setObjectName("RLPanelTitle")
        layout.addWidget(title)

        mode_hint = QLabel("Current Mode: Single-Scene Training")
        mode_hint.setObjectName("RLPanelHint")
        mode_hint.setWordWrap(True)
        layout.addWidget(mode_hint)

        # ---- 1. 算法选择 ----
        layout.addWidget(make_section_label("Training Algorithm"))
        self.algorithm_combo = QComboBox()
        for name in RL_ALGORITHMS:
            self.algorithm_combo.addItem(name, name)
        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        layout.addWidget(self.algorithm_combo)

        # ---- 2. 训练配置 ----
        layout.addWidget(make_section_label("Training Configuration"))
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 100000)
        self.episodes_spin.setSingleStep(50)
        self.episodes_spin.setValue(500)
        self.episodes_spin.setToolTip("Total episodes in one training run")
        layout.addLayout(self._param_row("Training Episodes", self.episodes_spin))

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(50, 100000)
        self.max_steps_spin.setSingleStep(50)
        self.max_steps_spin.setValue(1200)
        self.max_steps_spin.setToolTip(
            "Maximum steps per episode; longer episodes are truncated"
        )
        layout.addLayout(self._param_row("Max Steps per Episode", self.max_steps_spin))

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(0)
        self.seed_spin.setToolTip(
            "Training random seed; reuse the same configuration and seed "
            "for reproducible experiments"
        )
        layout.addLayout(self._param_row("Random Seed", self.seed_spin))

        self.generalize_check = QCheckBox("Generalized Training (All 6 Routes)")
        self.generalize_check.setChecked(False)
        self.generalize_check.setToolTip(
            "When enabled, each episode uses a random (entry, exit) route so "
            "the policy can learn all six routes. When disabled, training uses "
            "the single route selected below."
        )
        self.generalize_check.toggled.connect(self._on_generalize_toggled)
        layout.addWidget(self.generalize_check)

        # 单路训练的路线选择(泛化开启时禁用)
        self.train_entry_combo = QComboBox()
        self.train_exit_combo = QComboBox()
        layout.addLayout(self._param_row("Training Entry", self.train_entry_combo))
        layout.addLayout(self._param_row("Training Exit", self.train_exit_combo))

        # ---- 3. 训练控制 ----
        layout.addWidget(make_section_label("Training Controls"))
        ctrl_row = QHBoxLayout()
        ctrl_row.setContentsMargins(0, 0, 0, 0)
        ctrl_row.setSpacing(6)
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setObjectName("RLStartButton")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setObjectName("RLStopButton")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self.start_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addStretch(1)
        layout.addLayout(ctrl_row)

        # 状态显示
        self.status_label = QLabel("Status: Idle")
        self.status_label.setObjectName("RLStatusLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # ---- 3. 模型加载 ----
        layout.addWidget(make_section_label("Trained Model"))
        load_row = QHBoxLayout()
        load_row.setContentsMargins(0, 0, 0, 0)
        load_row.setSpacing(6)
        self.load_btn = make_step_button(
            "Load Checkpoint", "Choose a trained model file (.zip/.pt/.ckpt)"
        )
        self.load_btn.clicked.connect(self._on_load_model)
        load_row.addWidget(self.load_btn)
        load_row.addStretch(1)
        layout.addLayout(load_row)
        self.model_path_label = QLabel("Not loaded")
        self.model_path_label.setObjectName("RLPanelHint")
        self.model_path_label.setWordWrap(True)
        layout.addWidget(self.model_path_label)

        # ---- 推理路线 ----
        layout.addWidget(make_section_label("Inference Route"))
        self.follow_scene_check = QCheckBox("Follow Scene 1 Settings")
        self.follow_scene_check.setChecked(False)
        self.follow_scene_check.setToolTip(
            "Checked: use the (entry, exit) route selected in Scene 1 controls.\n"
            "Unchecked: select the inference route independently below."
        )
        self.follow_scene_check.toggled.connect(self._on_follow_scene_toggled)
        layout.addWidget(self.follow_scene_check)

        # 当前场景 1 路线提示(workspace 通过 set_scene_route_hint 喂)
        self.scene_route_hint = QLabel("Scene 1: —")
        self.scene_route_hint.setObjectName("RLPanelHint")
        self.scene_route_hint.setWordWrap(True)
        layout.addWidget(self.scene_route_hint)

        self.entry_combo = QComboBox()
        self.exit_combo = QComboBox()
        # 延迟到 set_route_presets() 才填充选项,避免在 panel 里硬编码 scene 数据
        layout.addLayout(self._param_row("Inference Entry", self.entry_combo))
        layout.addLayout(self._param_row("Inference Exit", self.exit_combo))
        self.entry_combo.setEnabled(True)
        self.exit_combo.setEnabled(True)

        # ---- 分段速度:训练和 UI 推理共用同一份条件 ----
        layout.addWidget(make_section_label("Segmented Speed (Training/Inference)"))
        self.speed_profile_combo = QComboBox()
        for name, description in (
            ("B0", "B0 Fixed 0.50 / 0.50 mm"),
            ("A1", "A1 0.75 / 0.50 mm"),
            ("A2", "A2 1.00 / 0.50 mm"),
            ("A3", "A3 0.75 / 0.25 mm"),
            ("A4", "A4 1.50 / 0.50 mm"),
            ("A5", "A5 2.00 / 0.50 mm"),
            ("A6", "A6 2.25 / 0.50 mm"),
            ("A7", "A7 2.50 / 0.50 mm"),
            ("P75", "P75 2.00 / 0.50 mm (switch at 0.75)"),
            ("P85", "P85 2.00 / 0.50 mm (switch at 0.85)"),
            ("P90", "P90 2.00 / 0.50 mm (switch at 0.90)"),
            ("P95", "P95 2.00 / 0.50 mm (switch at 0.95)"),
            ("CUSTOM", "Custom"),
        ):
            self.speed_profile_combo.addItem(description, name)
        self.speed_profile_combo.setToolTip(
            "Values before and after the slash are the advance distances before "
            "and after the centerline-progress switch. This changes ADVANCE only; "
            "retraction, rotation, rewards, and PPO parameters are unchanged."
        )
        layout.addLayout(self._param_row("Speed Profile", self.speed_profile_combo))

        self.proximal_push_spin = FocusWheelDoubleSpinBox()
        self.proximal_push_spin.setRange(0.05, 5.00)
        self.proximal_push_spin.setDecimals(2)
        self.proximal_push_spin.setSingleStep(0.05)
        self.distal_push_spin = FocusWheelDoubleSpinBox()
        self.distal_push_spin.setRange(0.05, 5.00)
        self.distal_push_spin.setDecimals(2)
        self.distal_push_spin.setSingleStep(0.05)
        self.speed_switch_spin = FocusWheelDoubleSpinBox()
        self.speed_switch_spin.setRange(0.00, 1.00)
        self.speed_switch_spin.setDecimals(2)
        self.speed_switch_spin.setSingleStep(0.01)
        self.speed_switch_spin.setToolTip(
            "Switch using normalized centerline arc-length progress, not the "
            "straight-line distance from the entry to the guidewire tip. Once "
            "the threshold is reached, distal slow speed remains active for "
            "the rest of the episode."
        )
        layout.addLayout(self._param_row("Proximal Advance (mm)", self.proximal_push_spin))
        layout.addLayout(self._param_row("Distal Advance (mm)", self.distal_push_spin))
        layout.addLayout(self._param_row("Switch Progress", self.speed_switch_spin))
        self.speed_profile_combo.currentIndexChanged.connect(
            self._on_speed_profile_changed
        )
        self._on_speed_profile_changed()

        # 运行模型(推理回放):用 ckpt 控制 UI 上的 wire 走到终点
        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0); run_row.setSpacing(6)
        self.run_btn = QPushButton("▶ Run Model")
        self.run_btn.setObjectName("RLStartButton")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.setEnabled(False)   # 未加载模型前禁用
        self.run_btn.clicked.connect(self._on_run_model)
        self.stop_run_btn = QPushButton("■ Stop Inference")
        self.stop_run_btn.setObjectName("RLStopButton")
        self.stop_run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_run_btn.setEnabled(False)
        self.stop_run_btn.clicked.connect(self._on_stop_run)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.stop_run_btn)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        # ---- 4. 奖励参数(折叠) ----
        self.reward_toggle_btn = QToolButton()
        self.reward_toggle_btn.setObjectName("RLToggleSection")
        self.reward_toggle_btn.setText("▶ Reward Parameters")
        self.reward_toggle_btn.setCheckable(True)
        self.reward_toggle_btn.setChecked(False)
        self.reward_toggle_btn.toggled.connect(self._on_reward_toggle)
        layout.addWidget(self.reward_toggle_btn)

        self.reward_widget = QFrame()
        self.reward_widget.setVisible(False)
        rl = QVBoxLayout(self.reward_widget)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        for label, key, decimals, step, vmin, vmax in self._reward_field_specs():
            spin = FocusWheelDoubleSpinBox()
            spin.setRange(vmin, vmax)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setValue(float(DEFAULT_REWARD_PARAMS[key]))
            spin.valueChanged.connect(self._on_reward_changed)
            self._reward_spins[key] = spin
            rl.addLayout(self._param_row(label, spin))
        # 重置按钮
        reset_row = QHBoxLayout()
        reset_row.setContentsMargins(0, 0, 0, 0)
        reset_row.setSpacing(6)
        self.reward_reset_btn = make_step_button(
            "Restore Defaults", "Restore all reward parameters to their defaults"
        )
        self.reward_reset_btn.clicked.connect(self._on_reset_rewards)
        reset_row.addWidget(self.reward_reset_btn)
        reset_row.addStretch(1)
        rl.addLayout(reset_row)
        layout.addWidget(self.reward_widget)

        layout.addStretch(1)

    @staticmethod
    def _reward_field_specs():
        # (label, key, decimals, step, min, max)
        return [
            ("Progress Reward Weight", "progress_weight",  1, 5.0,   0.0,  500.0),
            ("Backtrack Penalty",      "backtrack_weight", 1, 5.0,   0.0,  500.0),
            ("Step Cost",              "step_cost",        3, 0.005, 0.0,  1.0),
            ("Blocked Penalty",        "blocked_penalty",  2, 0.05,  0.0,  10.0),
            ("Retract Cost",           "retract_cost",     3, 0.005, 0.0,  1.0),
            ("Rotate Cost",            "rotate_cost",      3, 0.005, 0.0,  1.0),
            ("Yaw Soft Threshold (°)", "yaw_soft_thresh",  0, 1.0,   0.0,  90.0),
            ("Yaw Soft Weight",        "yaw_soft_weight",  3, 0.005, 0.0,  1.0),
            ("Safety Margin (mm)",     "safety_margin_mm", 2, 0.05,  0.0,  5.0),
            ("Safety Weight",          "safety_weight",    2, 0.05,  0.0,  5.0),
            ("Success Bonus",          "success_bonus",    0, 5.0,   0.0,  1000.0),
            ("Timeout Penalty",        "timeout_penalty",  0, 1.0,   0.0,  500.0),
            ("Success Progress",       "success_progress", 2, 0.01,  0.50, 1.0),
        ]

    def _param_row(self, label_text, widget):
        row = QVBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        lab = QLabel(label_text)
        lab.setStyleSheet("color: #94a3b8; font-size: 11px;")
        lab.setWordWrap(True)
        lab.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        row.addWidget(lab)
        row.addWidget(widget)
        return row

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------

    def _on_reward_toggle(self, on: bool):
        self.reward_widget.setVisible(bool(on))
        self.reward_toggle_btn.setText(
            "▼ Reward Parameters" if on else "▶ Reward Parameters"
        )

    def _on_algorithm_changed(self):
        if self._suspend:
            return
        name = str(self.algorithm_combo.currentData() or "")
        self.algorithm_changed.emit(name)

    def _on_start(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.set_status(f"Training ({self.current_algorithm()}) …")
        self.start_clicked.emit()

    def _on_stop(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.set_status("Stopped")
        self.stop_clicked.emit()

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a Trained Model",
            "",
            "Model Files (*.zip *.pt *.ckpt *.pth);;All Files (*)",
        )
        if not path:
            return
        from pathlib import Path
        self.model_path_label.setText(f"Loaded: {Path(path).name}")
        # 加载完才允许跑推理
        self.run_btn.setEnabled(True)
        self.load_model_requested.emit(path)

    def _on_run_model(self):
        self.run_btn.setEnabled(False)
        self.stop_run_btn.setEnabled(True)
        self.run_model_clicked.emit()

    def _on_stop_run(self):
        self.stop_run_btn.setEnabled(False)
        self.run_model_clicked  # noqa - documenting,无副作用
        self.stop_model_clicked.emit()

    def set_inference_active(self, active: bool):
        """推理结束时由 workspace 调,复位按钮状态。"""
        self.run_btn.setEnabled(not active)
        self.stop_run_btn.setEnabled(bool(active))

    def _on_reward_changed(self, _value: Optional[float] = None):
        if self._suspend:
            return
        self.reward_params_changed.emit(self.get_reward_params())

    def _on_speed_profile_changed(self, _index: Optional[int] = None):
        name = str(self.speed_profile_combo.currentData() or "B0")
        custom = name == "CUSTOM"
        if not custom:
            cfg = SPEED_PROFILE_PRESETS[name]
            self._suspend = True
            self.proximal_push_spin.setValue(cfg.proximal_push_mm)
            self.distal_push_spin.setValue(cfg.distal_push_mm)
            self.speed_switch_spin.setValue(cfg.switch_progress)
            self._suspend = False
        for spin in (
            self.proximal_push_spin,
            self.distal_push_spin,
            self.speed_switch_spin,
        ):
            spin.setEnabled(custom)

    def _on_reset_rewards(self):
        self._suspend = True
        for key, spin in self._reward_spins.items():
            spin.setValue(float(DEFAULT_REWARD_PARAMS[key]))
        self._suspend = False
        self.reward_params_changed.emit(self.get_reward_params())

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def current_algorithm(self) -> str:
        return str(self.algorithm_combo.currentData() or RL_ALGORITHMS[0])

    def get_reward_params(self) -> dict:
        return {k: float(s.value()) for k, s in self._reward_spins.items()}

    def get_speed_config(self) -> dict:
        """Return the shared training/inference segmented-speed condition."""
        profile = str(self.speed_profile_combo.currentData() or "B0")
        cfg = resolve_speed_config({
            "profile": profile,
            "proximal_push_mm": float(self.proximal_push_spin.value()),
            "distal_push_mm": float(self.distal_push_spin.value()),
            "switch_progress": float(self.speed_switch_spin.value()),
        })
        return cfg.to_dict()

    # ------------------------------------------------------------------
    # 推理路线
    # ------------------------------------------------------------------

    def set_route_presets(self, entry_labels: list, exit_labels: list):
        """workspace 启动时把 (entry, exit) 预设标签灌进来(推理 + 训练两套下拉)。"""
        self._suspend = True
        for combo in (self.entry_combo, self.train_entry_combo):
            combo.clear()
            for i, lab in enumerate(entry_labels):
                combo.addItem(lab, i)
        for combo in (self.exit_combo, self.train_exit_combo):
            combo.clear()
            for i, lab in enumerate(exit_labels):
                combo.addItem(lab, i)
        # 训练默认起点入口1、终点出口3(与场景默认一致)
        self.train_exit_combo.setCurrentIndex(min(2, self.train_exit_combo.count() - 1))
        self._suspend = False

    def set_scene_route_hint(self, text: str):
        """显示场景 1 当前路线(workspace 在 entry/exit 切换时调)。"""
        self.scene_route_hint.setText(f"Scene 1: {text}")

    def get_inference_route(self) -> dict:
        """返回推理路线设置;follow=True 时 entry_idx/exit_idx 仅作 hint。"""
        return {
            "follow_scene":  bool(self.follow_scene_check.isChecked()),
            "entry_idx":     int(self.entry_combo.currentData() or 0),
            "exit_idx":      int(self.exit_combo.currentData() or 0),
        }

    def _on_follow_scene_toggled(self, checked: bool):
        # follow ON → dropdown 禁用;follow OFF → 让用户挑
        self.entry_combo.setEnabled(not checked)
        self.exit_combo.setEnabled(not checked)

    def get_training_config(self) -> dict:
        return {
            "max_episodes":          int(self.episodes_spin.value()),
            "max_steps_per_episode": int(self.max_steps_spin.value()),
            "seed":                  int(self.seed_spin.value()),
            "generalize":            bool(self.generalize_check.isChecked()),
            "train_entry_idx":       int(self.train_entry_combo.currentData() or 0),
            "train_exit_idx":        int(self.train_exit_combo.currentData() or 0),
            "speed_config":          self.get_speed_config(),
        }

    def _on_generalize_toggled(self, checked: bool):
        # 泛化 ON → 单路训练路线下拉禁用(随机抽);OFF → 让用户挑单条路线
        self.train_entry_combo.setEnabled(not checked)
        self.train_exit_combo.setEnabled(not checked)

    def set_status(self, text: str):
        self.status_label.setText(f"Status: {text}")

    def set_training_active(self, active: bool):
        """外部强制同步训练状态(例如后端报训练完毕)。"""
        self.start_btn.setEnabled(not active)
        self.stop_btn.setEnabled(bool(active))
