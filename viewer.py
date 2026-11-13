# 职责：独立 STL 查看器调试入口，用于单窗口检查 STL 模型加载和 PyVista 渲染效果。
import sys
from pathlib import Path

import pyvista as pv
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor

from zmd.utils.paths import ROOT_DIR, VESSELS_DIR, ROBOTS_DIR


DEFAULT_MODELS = [VESSELS_DIR / "VascuModel.stl", ROBOTS_DIR / "VascubotModel.stl"]

APP_STYLE = """
QMainWindow { background: #0f172a; }
QWidget { color: #e5e7eb; font-family: "Segoe UI", "Microsoft YaHei UI", Arial; font-size: 14px; }
#Header { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e3a8a, stop:1 #0891b2); border-radius: 18px; }
#Title { color: white; font-size: 26px; font-weight: 700; }
#Subtitle { color: #c7d2fe; font-size: 13px; }
#SidePanel, #ViewerPanel { background: #111827; border: 1px solid #243047; border-radius: 18px; }
#SectionTitle { color: #f8fafc; font-size: 16px; font-weight: 700; }
#MutedLabel { color: #94a3b8; font-size: 12px; font-weight: 600; }
#InfoCard { background: #172033; border: 1px solid #2d3b55; border-radius: 14px; }
#InfoValue { color: #f8fafc; font-size: 15px; font-weight: 700; }
#StatusLabel { color: #a7f3d0; background: #064e3b; border: 1px solid #047857; border-radius: 12px; padding: 10px 12px; }
QComboBox { background: #0b1220; border: 1px solid #334155; border-radius: 12px; padding: 10px 12px; color: #f8fafc; }
QComboBox:hover { border-color: #38bdf8; }
QComboBox::drop-down { border: none; width: 28px; }
QComboBox QAbstractItemView { background: #111827; border: 1px solid #334155; color: #e5e7eb; selection-background-color: #2563eb; }
QPushButton { background: #2563eb; border: none; border-radius: 12px; color: white; font-weight: 700; padding: 11px 14px; }
QPushButton:hover { background: #1d4ed8; }
QPushButton:pressed { background: #1e40af; }
#SecondaryButton { background: #334155; }
#SecondaryButton:hover { background: #475569; }
"""


class StlViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZMD STL 3D Viewer")
        self.resize(1320, 860)
        self.setMinimumSize(980, 680)
        self.plotter = QtInteractor(self)
        self.model_selector = QComboBox()
        self.status_label = QLabel("Ready"); self.status_label.setObjectName("StatusLabel")
        self.file_value = QLabel("No model"); self.file_value.setObjectName("InfoValue")
        self.points_value = QLabel("-"); self.points_value.setObjectName("InfoValue")
        self.cells_value = QLabel("-"); self.cells_value.setObjectName("InfoValue")
        for model_path in DEFAULT_MODELS:
            if model_path.exists(): self.model_selector.addItem(model_path.name, model_path)
        self.model_selector.currentIndexChanged.connect(self.load_selected_model)
        self.setCentralWidget(self.build_ui()); self.configure_plotter()
        if self.model_selector.count() > 0: self.load_selected_model()
        else: self.status_label.setText("No STL files found. Click Open STL to load one.")

    def build_ui(self):
        root = QWidget(); root.setStyleSheet(APP_STYLE)
        root_layout = QVBoxLayout(root); root_layout.setContentsMargins(20, 20, 20, 20); root_layout.setSpacing(16)
        header = QFrame(); header.setObjectName("Header")
        header_layout = QVBoxLayout(header); header_layout.setContentsMargins(24, 18, 24, 18); header_layout.setSpacing(4)
        title = QLabel("ZMD STL 3D Viewer"); title.setObjectName("Title")
        subtitle = QLabel("Interactive vascular model visualization workspace"); subtitle.setObjectName("Subtitle")
        header_layout.addWidget(title); header_layout.addWidget(subtitle)
        content = QHBoxLayout(); content.setSpacing(16)
        content.addWidget(self.build_side_panel()); content.addWidget(self.build_viewer_panel(), 1)
        root_layout.addWidget(header); root_layout.addLayout(content, 1); return root

    def build_side_panel(self):
        panel = QFrame(); panel.setObjectName("SidePanel"); panel.setFixedWidth(320)
        layout = QVBoxLayout(panel); layout.setContentsMargins(20, 20, 20, 20); layout.setSpacing(16)
        section = QLabel("Model Controls"); section.setObjectName("SectionTitle")
        model_label = QLabel("SELECT MODEL"); model_label.setObjectName("MutedLabel")
        open_button = QPushButton("Open STL File"); open_button.clicked.connect(self.open_model)
        reset_button = QPushButton("Reset View"); reset_button.setObjectName("SecondaryButton"); reset_button.clicked.connect(self.plotter.reset_camera)
        layout.addWidget(section); layout.addWidget(model_label); layout.addWidget(self.model_selector)
        layout.addWidget(open_button); layout.addWidget(reset_button); layout.addSpacing(6)
        layout.addWidget(self.make_info_card("CURRENT FILE", self.file_value))
        layout.addWidget(self.make_info_card("POINTS", self.points_value))
        layout.addWidget(self.make_info_card("CELLS", self.cells_value))
        layout.addStretch(1); layout.addWidget(self.status_label); return panel

    def build_viewer_panel(self):
        panel = QFrame(); panel.setObjectName("ViewerPanel")
        layout = QVBoxLayout(panel); layout.setContentsMargins(14, 14, 14, 14); layout.setSpacing(0)
        self.plotter.interactor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.plotter.interactor, 1); return panel

    def make_info_card(self, label_text, value_widget):
        card = QFrame(); card.setObjectName("InfoCard")
        layout = QVBoxLayout(card); layout.setContentsMargins(14, 12, 14, 12); layout.setSpacing(5)
        label = QLabel(label_text); label.setObjectName("MutedLabel")
        value_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); value_widget.setWordWrap(True)
        layout.addWidget(label); layout.addWidget(value_widget); return card

    def configure_plotter(self):
        self.plotter.set_background("#0b1020", top="#18243a"); self.plotter.add_axes(color="white"); self.plotter.enable_anti_aliasing()

    def load_selected_model(self):
        model_path = self.model_selector.currentData()
        if model_path: self.load_model(Path(model_path))

    def open_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open STL model", str(VESSELS_DIR), "STL files (*.stl);;All files (*)")
        if not file_name: return
        model_path = Path(file_name)
        existing_index = self.model_selector.findData(model_path)
        if existing_index >= 0: self.model_selector.setCurrentIndex(existing_index); return
        self.model_selector.addItem(model_path.name, model_path); self.model_selector.setCurrentIndex(self.model_selector.count() - 1)

    def load_model(self, model_path: Path):
        self.status_label.setText(f"Loading {model_path.name}..."); QApplication.processEvents()
        mesh = pv.read(model_path); self.plotter.clear(); self.plotter.set_background("#0b1020", top="#18243a")
        self.plotter.add_mesh(mesh, color="#dbeafe", specular=0.35, specular_power=20, smooth_shading=True)
        self.plotter.add_axes(color="white"); self.plotter.reset_camera()
        self.file_value.setText(model_path.name); self.points_value.setText(f"{mesh.n_points:,}"); self.cells_value.setText(f"{mesh.n_cells:,}")
        self.status_label.setText("Model loaded successfully")


def main():
    app = QApplication(sys.argv)
    window = StlViewer(); window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
