# 职责：应用启动入口，仅创建 QApplication 并启动主窗口。
import sys

from PyQt6.QtWidgets import QApplication

from zmd.app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
