# 职责：首页界面，展示工程入口并向应用层发出用户意图信号。
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QLinearGradient, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zmd.ui.widgets import WINDOW_CONTROLS_STYLE, WindowControls


DESIGN_WIDTH = 1360
DESIGN_HEIGHT = 860


HOME_STYLE = f"""
QWidget {{
    background: transparent;
    color: #eef6ff;
    font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
}}
{WINDOW_CONTROLS_STYLE}
#BrandLine, #BrandAccent {{
    color: #ffffff;
    font-size: 42px;
    font-weight: 900;
}}
#BrandAccent {{
    color: #ffffff;
}}
#Subtitle {{
    color: #94a3b8;
    font-size: 18px;
}}
#Divider {{
    background: #1e3a66;
    max-height: 1px;
}}
#ProjectCard {{
    background: rgba(10, 35, 80, 0.62);
    border: 1px solid rgba(59, 130, 246, 0.75);
    border-radius: 18px;
}}
#CustomCard {{
    background: rgba(42, 22, 80, 0.62);
    border: 1px solid rgba(147, 51, 234, 0.75);
    border-radius: 18px;
}}
#IconCircleBlue {{
    background: rgba(37, 99, 235, 0.16);
    border: 2px solid #1d4ed8;
    border-radius: 48px;
    color: #eaf2ff;
    font-size: 44px;
    font-weight: 800;
}}
#IconCirclePurple {{
    background: rgba(126, 34, 206, 0.18);
    border: 2px solid #6d28d9;
    border-radius: 48px;
    color: #ffffff;
    font-size: 42px;
    font-weight: 800;
}}
#CardTitle {{
    color: #ffffff;
    font-size: 25px;
    font-weight: 900;
}}
#CardDescription {{
    color: #9caac0;
    font-size: 15px;
    line-height: 1.5;
}}
#FileRow {{
    background: rgba(15, 23, 42, 0.68);
    border: 1px solid #3d4b66;
    border-radius: 10px;
}}
#FileLabel {{
    color: #dbe7f6;
    font-size: 14px;
    font-weight: 700;
}}
#FileValue {{
    color: #94a3b8;
    font-size: 12px;
    font-weight: 700;
}}
#ChooseFileButton {{
    background: rgba(30, 41, 69, 0.9);
    border: 1px solid #4b5f83;
    border-radius: 8px;
    color: #ffffff;
    font-size: 13px;
    font-weight: 800;
    padding: 8px 14px;
}}
#ChooseFileButton:disabled {{
    color: rgba(255, 255, 255, 0.55);
}}
#PrimaryButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #2563eb);
    border: none;
    border-radius: 10px;
    color: white;
    font-size: 17px;
    font-weight: 900;
    padding: 15px 28px;
}}
#PrimaryButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #60a5fa, stop:1 #2563eb);
}}
#PurpleButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9333ea, stop:1 #6d28d9);
    border: none;
    border-radius: 10px;
    color: white;
    font-size: 17px;
    font-weight: 900;
    padding: 15px 28px;
}}
#PurpleButton:disabled {{
    color: rgba(255, 255, 255, 0.75);
    background: rgba(126, 34, 206, 0.55);
}}
"""


class HomePage(QGraphicsView):
    close_requested = pyqtSignal()
    minimize_requested = pyqtSignal()
    fullscreen_requested = pyqtSignal()
    enter_default_workspace_requested = pyqtSignal()
    enter_custom_workspace_requested = pyqtSignal()
    vascular_model_select_requested = pyqtSignal()
    robot_model_select_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.drag_position = None
        self.content = HomeContent()
        self.scene = QGraphicsScene(self)
        self.proxy = self.scene.addWidget(self.content)
        self.setScene(self.scene)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setStyleSheet("background: transparent; border: none;")
        self.viewport().setStyleSheet("background: transparent;")
        self.connect_content_signals()

    def connect_content_signals(self):
        self.content.close_requested.connect(self.close_requested.emit)
        self.content.minimize_requested.connect(self.minimize_requested.emit)
        self.content.fullscreen_requested.connect(self.fullscreen_requested.emit)
        self.content.enter_default_workspace_requested.connect(self.enter_default_workspace_requested.emit)
        self.content.enter_custom_workspace_requested.connect(self.enter_custom_workspace_requested.emit)
        self.content.vascular_model_select_requested.connect(self.vascular_model_select_requested.emit)
        self.content.robot_model_select_requested.connect(self.robot_model_select_requested.emit)

    def set_fullscreen_state(self, is_fullscreen):
        self.content.set_fullscreen_state(is_fullscreen)

    def set_custom_vascular_model_path(self, model_path):
        self.content.set_vascular_model_path(model_path)

    def set_custom_robot_model_path(self, model_path):
        self.content.set_robot_model_path(model_path)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        viewport_width = max(1, self.viewport().width())
        viewport_height = max(1, self.viewport().height())
        scale = min(viewport_width / DESIGN_WIDTH, viewport_height / DESIGN_HEIGHT)
        scene_width = viewport_width / scale
        scene_height = viewport_height / scale
        self.scene.setSceneRect(0, 0, scene_width, scene_height)
        self.resetTransform()
        self.scale(scale, scale)
        self.proxy.setPos((scene_width - DESIGN_WIDTH) / 2, (scene_height - DESIGN_HEIGHT) / 2)

    def drawBackground(self, painter, rect):
        scene_rect = self.scene.sceneRect()
        gradient = QLinearGradient(scene_rect.topLeft(), scene_rect.bottomRight())
        gradient.setColorAt(0.0, QColor("#061124"))
        gradient.setColorAt(0.45, QColor("#07101f"))
        gradient.setColorAt(1.0, QColor("#120728"))
        painter.fillRect(scene_rect, QBrush(gradient))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(37, 99, 235, 60))
        for i in range(12):
            y = int(scene_rect.top() + scene_rect.height() * 0.55 + i * 9)
            painter.drawArc(int(scene_rect.left() - 180), y, 520, 180, 0, 180 * 16)
        painter.setPen(QColor(126, 34, 206, 45))
        for i in range(10):
            y = int(scene_rect.top() + scene_rect.height() * 0.58 + i * 10)
            painter.drawArc(int(scene_rect.right() - 360), y, 520, 160, 0, 180 * 16)

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
        scene_position = self.mapToScene(position)
        content_position = self.proxy.mapFromScene(scene_position).toPoint()
        if not self.content.rect().contains(content_position):
            return False
        widget = self.content.childAt(content_position)
        while widget:
            if isinstance(widget, QPushButton):
                return True
            widget = widget.parentWidget()
        return False


class HomeContent(QWidget):
    close_requested = pyqtSignal()
    minimize_requested = pyqtSignal()
    fullscreen_requested = pyqtSignal()
    enter_default_workspace_requested = pyqtSignal()
    enter_custom_workspace_requested = pyqtSignal()
    vascular_model_select_requested = pyqtSignal()
    robot_model_select_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.window_controls = WindowControls()
        self.custom_workspace_button = None
        self.vascular_file_value = None
        self.robot_file_value = None
        self.setFixedSize(DESIGN_WIDTH, DESIGN_HEIGHT)
        self.setStyleSheet(HOME_STYLE)
        self.build_ui()
        self.connect_window_controls()

    def connect_window_controls(self):
        self.window_controls.close_requested.connect(self.close_requested.emit)
        self.window_controls.minimize_requested.connect(self.minimize_requested.emit)
        self.window_controls.fullscreen_requested.connect(self.fullscreen_requested.emit)

    def set_fullscreen_state(self, is_fullscreen):
        self.window_controls.set_fullscreen_state(is_fullscreen)

    def set_vascular_model_path(self, model_path):
        self.update_file_value(self.vascular_file_value, model_path)
        self.custom_workspace_button.setEnabled(bool(model_path))

    def set_robot_model_path(self, model_path):
        self.update_file_value(
            self.robot_file_value,
            model_path,
            default_text="Not selected; the default robot model will be used",
        )

    def update_file_value(self, label, model_path, default_text="Not selected"):
        if not label:
            return
        if not model_path:
            label.setText(default_text)
            label.setToolTip(default_text)
            return
        path = Path(model_path)
        label.setText(path.name)
        label.setToolTip(str(path))

    def build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(28, 18, 28, 42)
        root_layout.setSpacing(0)
        root_layout.addLayout(self.build_window_controls())
        root_layout.addStretch(1)
        root_layout.addLayout(self.build_title_area())
        root_layout.addSpacing(34)
        root_layout.addLayout(self.build_project_area())
        root_layout.addStretch(1)

    def build_window_controls(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)
        layout.addWidget(self.window_controls)
        return layout

    def build_title_area(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(14)
        brand_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_accent = QLabel("Medical Visualization")
        brand_accent.setObjectName("BrandAccent")
        brand_text = QLabel("3D Modeling Workspace")
        brand_text.setObjectName("BrandLine")
        brand_layout.addWidget(brand_accent)
        brand_layout.addWidget(brand_text)
        subtitle_row = QHBoxLayout()
        subtitle_row.setSpacing(28)
        subtitle_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_row.addWidget(self.make_divider())
        subtitle = QLabel("AI-Powered Training Platform for Endovascular Intervention")
        subtitle.setObjectName("Subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_row.addWidget(subtitle)
        subtitle_row.addWidget(self.make_divider())
        layout.addLayout(brand_layout)
        layout.addLayout(subtitle_row)
        return layout

    def build_project_area(self):
        layout = QHBoxLayout()
        layout.setSpacing(24)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.build_default_card())
        layout.addWidget(self.build_custom_card())
        return layout

    def build_default_card(self):
        card = self.make_card("ProjectCard")
        layout = card.layout()
        layout.addSpacing(28)
        layout.addWidget(self.make_icon_circle("▰", "IconCircleBlue"), alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(22)
        layout.addWidget(self.make_label("Default Project", "CardTitle"), alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(8)
        description = self.make_label("Open the built-in sample project", "CardDescription")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)
        layout.addSpacing(18)
        placeholder = QWidget()
        placeholder.setFixedHeight(106)
        layout.addWidget(placeholder)
        layout.addStretch(1)
        button = QPushButton("🚀  Open Workspace")
        button.setObjectName("PrimaryButton")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(self.enter_default_workspace_requested.emit)
        layout.addWidget(button)
        return card

    def build_custom_card(self):
        card = self.make_card("CustomCard")
        layout = card.layout()
        layout.addSpacing(28)
        layout.addWidget(self.make_icon_circle("▣", "IconCirclePurple"), alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(22)
        layout.addWidget(self.make_label("Custom Project", "CardTitle"), alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(8)
        description = self.make_label(
            "Import a vessel model (STL)\nand an optional robot model",
            "CardDescription",
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)
        layout.addSpacing(18)
        vascular_row, self.vascular_file_value = self.make_file_row(
            "Vessel Model (STL)",
            self.vascular_model_select_requested.emit,
            enabled=True,
            default_text="Not selected",
        )
        robot_row, self.robot_file_value = self.make_file_row(
            "Robot Model (STL, optional)",
            self.robot_model_select_requested.emit,
            enabled=True,
            default_text="Not selected; the default robot model will be used",
        )
        layout.addWidget(vascular_row)
        layout.addWidget(robot_row)
        layout.addStretch(1)
        self.custom_workspace_button = QPushButton("→  Open Workspace")
        self.custom_workspace_button.setObjectName("PurpleButton")
        self.custom_workspace_button.setEnabled(False)
        self.custom_workspace_button.clicked.connect(self.enter_custom_workspace_requested.emit)
        layout.addWidget(self.custom_workspace_button)
        return card

    def make_card(self, object_name):
        card = QFrame()
        card.setObjectName(object_name)
        card.setFixedSize(430, 520)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(36, 36, 36, 32)
        layout.setSpacing(0)
        return card

    def make_icon_circle(self, text, object_name):
        label = QLabel(text)
        label.setObjectName(object_name)
        label.setFixedSize(96, 96)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def make_file_row(self, label_text, handler, enabled, default_text):
        row = QFrame()
        row.setObjectName("FileRow")
        row.setFixedHeight(58)
        layout = QGridLayout(row)
        layout.setContentsMargins(14, 6, 8, 6)
        layout.setColumnStretch(0, 1)
        label = QLabel(label_text)
        label.setObjectName("FileLabel")
        value_label = QLabel(default_text)
        value_label.setObjectName("FileValue")
        value_label.setToolTip(default_text)
        choose_button = QPushButton("▣  Choose File")
        choose_button.setObjectName("ChooseFileButton")
        choose_button.setCursor(Qt.CursorShape.PointingHandCursor)
        choose_button.setEnabled(enabled)
        choose_button.clicked.connect(handler)
        layout.addWidget(label, 0, 0)
        layout.addWidget(value_label, 1, 0)
        layout.addWidget(choose_button, 0, 1, 2, 1)
        return row, value_label

    def make_label(self, text, object_name):
        label = QLabel(text)
        label.setObjectName(object_name)
        return label

    def make_divider(self):
        divider = QFrame()
        divider.setObjectName("Divider")
        divider.setFixedSize(100, 1)
        return divider
