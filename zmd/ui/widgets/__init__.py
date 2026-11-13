# 职责：可复用的无边框窗口控制按钮组件。
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget


WINDOW_CONTROLS_STYLE = """
#WindowControlButton, #WindowCloseButton {
    background: rgba(15, 23, 42, 0.42);
    border: 1px solid rgba(148, 163, 184, 0.38);
    border-radius: 16px;
    color: #dbe7f6;
    font-size: 20px;
    font-weight: 400;
    min-width: 34px;
    max-width: 34px;
    min-height: 34px;
    max-height: 34px;
    padding: 0;
}
#WindowControlButton:hover {
    background: rgba(59, 130, 246, 0.72);
    border-color: rgba(147, 197, 253, 0.86);
    color: #ffffff;
}
#WindowCloseButton {
    font-size: 24px;
    font-weight: 300;
}
#WindowCloseButton:hover {
    background: rgba(239, 68, 68, 0.88);
    border-color: rgba(252, 165, 165, 0.9);
    color: #ffffff;
}
"""


class WindowControls(QWidget):
    minimize_requested = pyqtSignal()
    fullscreen_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.fullscreen_button = self.make_button("□", "WindowControlButton")
        self.build_ui()

    def set_fullscreen_state(self, is_fullscreen):
        self.fullscreen_button.setText("❐" if is_fullscreen else "□")

    def build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        minimize_button = self.make_button("−", "WindowControlButton")
        minimize_button.clicked.connect(self.minimize_requested.emit)
        layout.addWidget(minimize_button)

        self.fullscreen_button.clicked.connect(self.fullscreen_requested.emit)
        layout.addWidget(self.fullscreen_button)

        close_button = self.make_button("×", "WindowCloseButton")
        close_button.clicked.connect(self.close_requested.emit)
        layout.addWidget(close_button)

    def make_button(self, text, object_name):
        button = QPushButton(text)
        button.setObjectName(object_name)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button
