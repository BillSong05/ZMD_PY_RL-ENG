# 职责：定义应用主窗口，管理页面栈、窗口状态和页面切换。
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from zmd.app.signal_router import SignalRouter
from zmd.io.project_loader import ProjectConfig, make_default_project_config
from zmd.ui.pages.home_page import HomePage
from zmd.ui.pages.workspace_page import WorkspacePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.normal_geometry = None

        self.home_page = HomePage()
        # 默认工作台提前创建（启动时在后台初始化，避免首次进入时卡顿）
        _default_config = make_default_project_config()
        self.workspace_page = WorkspacePage(_default_config)
        self._default_workspace_config_name = _default_config.name

        self.page_stack = QStackedWidget()
        self.page_stack.addWidget(self.home_page)
        self.page_stack.addWidget(self.workspace_page)
        self.current_page = self.home_page

        self.signal_router = SignalRouter(self)

        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setStyleSheet("background: #061124;")
        self.setWindowTitle("ZMD Visual Workspace")
        self.setMinimumSize(1680, 1080)
        self.resize(1680, 1080)
        self._center_on_primary_screen()

        self.signal_router.connect()
        self.signal_router.connect_workspace(self.workspace_page)
        self.setCentralWidget(self.page_stack)

    def _center_on_primary_screen(self):
        """把无边框窗口居中到主显示器。

        无边框窗口没有系统标题栏,Qt 默认放置在多屏环境下可能落到非主显示器
        (甚至被其它窗口挡住),导致"启动后看不到窗口"。这里强制以主屏可用区域
        居中,保证启动即可见。
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(avail.center())
        self.move(frame.topLeft())

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            if self.normal_geometry:
                self.setGeometry(self.normal_geometry)
            self.current_page.set_fullscreen_state(False)
            return
        self.normal_geometry = self.geometry()
        self.showFullScreen()
        self.current_page.set_fullscreen_state(True)

    def show_workspace(self, project_config: ProjectConfig):
        """进入工作台。默认工程直接复用已有实例；自定义工程重新创建。"""
        is_default = (project_config.name == self._default_workspace_config_name)
        if is_default and self.workspace_page is not None:
            # 直接复用预创建的默认工作台
            self.show_page(self.workspace_page)
            return
        # 自定义工程：替换旧工作台（如果旧的是默认工作台则保留，否则销毁）
        if (self.workspace_page is not None and
                self.workspace_page.project_config.name != self._default_workspace_config_name):
            self.page_stack.removeWidget(self.workspace_page)
            self.workspace_page.deleteLater()
        new_page = WorkspacePage(project_config)
        self.signal_router.connect_workspace(new_page)
        self.page_stack.addWidget(new_page)
        # 如果旧工作台是默认的，把它藏起来但不删，保留备用
        self.workspace_page = new_page
        self.show_page(self.workspace_page)

    def show_home_page(self):
        """返回首页 — 只切换页面，不销毁工作台（保留物理/渲染状态）。"""
        self.show_page(self.home_page)

    def show_page(self, page):
        self.current_page = page
        self.current_page.set_fullscreen_state(self.isFullScreen())
        self.page_stack.setCurrentWidget(self.current_page)

    def closeEvent(self, event):
        seen = set()
        for page in (getattr(self, "workspace_page", None), getattr(self, "current_page", None)):
            if page is None or id(page) in seen:
                continue
            seen.add(id(page))
            shutdown = getattr(page, "shutdown", None)
            if callable(shutdown):
                shutdown()
        super().closeEvent(event)
