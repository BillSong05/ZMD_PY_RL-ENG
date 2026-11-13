# 职责：集中连接界面信号与应用层处理函数。
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from zmd.io.project_loader import make_default_project_config, ProjectConfig


class SignalRouter:
    def __init__(self, window):
        self.window = window
        self.custom_vascular_model_path = None
        self.custom_robot_model_path = None

    def connect(self):
        home_page = self.window.home_page
        home_page.close_requested.connect(self.window.close)
        home_page.minimize_requested.connect(self.window.showMinimized)
        home_page.fullscreen_requested.connect(self.window.toggle_fullscreen)
        home_page.enter_default_workspace_requested.connect(self.enter_default_workspace)
        home_page.enter_custom_workspace_requested.connect(self.enter_custom_workspace)
        home_page.vascular_model_select_requested.connect(self.select_vascular_model)
        home_page.robot_model_select_requested.connect(self.select_robot_model)

    def connect_workspace(self, workspace_page):
        workspace_page.return_home_requested.connect(self.return_home)
        workspace_page.close_requested.connect(self.window.close)
        workspace_page.minimize_requested.connect(self.window.showMinimized)
        workspace_page.fullscreen_requested.connect(self.window.toggle_fullscreen)

    def enter_default_workspace(self):
        config = make_default_project_config()
        self.window.show_workspace(config)

    def return_home(self):
        self.window.show_home_page()

    def enter_custom_workspace(self):
        if not self.custom_vascular_model_path:
            QMessageBox.warning(
                self.window,
                "Vessel Model Required",
                "Select a vessel model STL file first.",
            )
            return
        config = ProjectConfig(
            name="custom_project",
            vessel_model_path=Path(self.custom_vascular_model_path),
            robot_model_path=Path(self.custom_robot_model_path) if self.custom_robot_model_path else None,
            scene_layout="four_view",
            compact_vascular_controls=False,
        )
        self.window.show_workspace(config)

    def select_vascular_model(self):
        model_path = self.choose_stl_model(
            "Choose Vessel Model STL", self.custom_vascular_model_path
        )
        if not model_path:
            return
        self.custom_vascular_model_path = model_path
        self.window.home_page.set_custom_vascular_model_path(model_path)

    def select_robot_model(self):
        model_path = self.choose_stl_model(
            "Choose Robot Model STL", self.custom_robot_model_path
        )
        if not model_path:
            return
        self.custom_robot_model_path = model_path
        self.window.home_page.set_custom_robot_model_path(model_path)

    def choose_stl_model(self, title, current_path=None):
        start_dir = str(Path(current_path).parent) if current_path else str(Path.cwd())
        file_name, _ = QFileDialog.getOpenFileName(
            self.window,
            title,
            start_dir,
            "STL files (*.stl);;All files (*)",
        )
        if not file_name:
            return None
        path = Path(file_name)
        if not path.exists() or path.suffix.lower() != ".stl":
            QMessageBox.warning(
                self.window,
                "Invalid File",
                "Select a valid STL model file.",
            )
            return None
        return path
