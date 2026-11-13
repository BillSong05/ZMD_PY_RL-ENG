# 职责：默认工作台底部调试信息面板，集中展示场景状态、训练曲线、硬件占用和算法对比信息。
from collections import deque
import ctypes
import shutil
import subprocess

from PyQt6.QtCore import QDateTime, QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget


SCENE_COLORS = ("#60a5fa", "#34d399", "#fbbf24", "#f472b6")
ALGORITHM_COLORS = ("#60a5fa", "#34d399", "#f59e0b", "#f472b6")

DEBUG_PANEL_STYLE = """
#DebugPanel {
    background: rgba(10, 35, 80, 0.58);
    border: 1px solid rgba(59, 130, 246, 0.56);
    border-radius: 14px;
}
#DebugPanelTitle {
    color: #dbeafe;
    font-size: 13px;
    font-weight: 900;
}
#DebugPanelClearButton, #DebugSlotSwitchButton {
    background: rgba(15, 23, 42, 0.66);
    border: 1px solid rgba(148, 163, 184, 0.42);
    border-radius: 9px;
    color: #dbe7f6;
    font-size: 11px;
    font-weight: 800;
    padding: 3px 8px;
}
#DebugPanelClearButton:hover, #DebugSlotSwitchButton:hover {
    background: rgba(59, 130, 246, 0.78);
    border-color: rgba(147, 197, 253, 0.88);
    color: #ffffff;
}
#DebugSlotSwitchButton:checked {
    background: rgba(37, 99, 235, 0.86);
    border-color: rgba(147, 197, 253, 0.92);
    color: #ffffff;
}
#DebugSection {
    background: rgba(2, 6, 23, 0.28);
    border: 1px solid rgba(30, 64, 175, 0.42);
    border-radius: 10px;
}
#DebugSectionTitle {
    color: #dbeafe;
    font-size: 12px;
    font-weight: 800;
}
#DebugPanelLog {
    background: rgba(2, 6, 23, 0.30);
    border: none;
    color: #c7d2fe;
    font-family: "Cascadia Mono", Consolas, monospace;
    font-size: 11px;
    padding: 4px;
}
#DebugPanelLog QScrollBar:vertical {
    background: rgba(15, 23, 42, 0.36);
    border: none;
    border-radius: 5px;
    width: 10px;
}
#DebugPanelLog QScrollBar::handle:vertical {
    background: rgba(96, 165, 250, 0.78);
    border-radius: 5px;
    min-height: 24px;
}
#DebugPanelLog QScrollBar::add-line:vertical, #DebugPanelLog QScrollBar::sub-line:vertical,
#DebugPanelLog QScrollBar::add-page:vertical, #DebugPanelLog QScrollBar::sub-page:vertical {
    background: transparent;
    border: none;
    height: 0;
}
#HardwareMetricLabel {
    color: #dbe7f6;
    font-size: 11px;
    font-weight: 800;
}
#HardwareMetricValue {
    color: #bfdbfe;
    font-size: 11px;
    font-weight: 800;
}
#HardwareMetricBar {
    background: rgba(15, 23, 42, 0.58);
    border: 1px solid rgba(59, 130, 246, 0.32);
    border-radius: 5px;
    height: 10px;
}
#HardwareMetricBar::chunk {
    background: rgba(96, 165, 250, 0.88);
    border-radius: 4px;
}
"""


class DebugSection(QFrame):
    def __init__(self, title, widget, actions=None):
        super().__init__()
        self.setObjectName("DebugSection")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 7, 8, 8)
        layout.setSpacing(5)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("DebugSectionTitle")
        title_row.addWidget(self.title_label)
        title_row.addStretch(1)
        for action in actions or []:
            title_row.addWidget(action)
        layout.addLayout(title_row)
        layout.addWidget(widget, 1)

    def set_title(self, title: str):
        self.title_label.setText(title)


class SceneCurveWidget(QWidget):
    MAX_POINTS = 160
    def __init__(self, empty_text, fixed_range=None, suffix=""):
        super().__init__()
        self.empty_text = empty_text
        self.fixed_range = fixed_range
        self.suffix = suffix
        self.scene_values = {index: deque(maxlen=self.MAX_POINTS) for index in range(4)}
        self.single_scene_mode = False
        self.x_axis_label = "Training Iterations"
        self.setMinimumHeight(100)

    def set_single_scene_mode(self, single: bool):
        """单场景模式只画/列 scene 0。"""
        self.single_scene_mode = bool(single)
        self.update()

    def set_x_axis_label(self, text: str):
        self.x_axis_label = str(text) if text else "Training Iterations"
        self.update()

    def append_value(self, scene_index, value):
        if scene_index not in self.scene_values:
            return
        self.scene_values[scene_index].append(float(value))
        self.update()

    def set_scene_values(self, scene_index, values):
        if scene_index not in self.scene_values:
            return
        self.scene_values[scene_index] = deque(list(values)[-self.MAX_POINTS:], maxlen=self.MAX_POINTS)
        self.update()

    def clear_values(self):
        for values in self.scene_values.values():
            values.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(2, 6, 23, 44))
        chart_rect = QRectF(34.0, 14.0, max(1.0, self.width() - 48.0), max(1.0, self.height() - 46.0))
        self.draw_grid(painter, chart_rect)
        all_values = [value for values in self.scene_values.values() for value in values]
        if not all_values:
            painter.setPen(QColor("#64748b"))
            painter.drawText(chart_rect, Qt.AlignmentFlag.AlignCenter, self.empty_text)
            self.draw_x_axis_label(painter, chart_rect)
            self.draw_legend(painter)
            return
        minimum, maximum = self.value_range(all_values)
        self.draw_axis_labels(painter, chart_rect, minimum, maximum)
        for scene_index, values in self.scene_values.items():
            if self.single_scene_mode and scene_index != 0:
                continue
            if len(values) < 2:
                continue
            self.draw_line(painter, chart_rect, scene_index, list(values), minimum, maximum)
        self.draw_x_axis_label(painter, chart_rect)
        self.draw_legend(painter)

    def draw_grid(self, painter, chart_rect):
        painter.setPen(QPen(QColor(59, 130, 246, 46), 1, Qt.PenStyle.DotLine))
        for step in range(5):
            y = chart_rect.top() + chart_rect.height() * step / 4
            painter.drawLine(QPointF(chart_rect.left(), y), QPointF(chart_rect.right(), y))
        for step in range(6):
            x = chart_rect.left() + chart_rect.width() * step / 5
            painter.drawLine(QPointF(x, chart_rect.top()), QPointF(x, chart_rect.bottom()))
        painter.setPen(QPen(QColor(96, 165, 250, 78), 1))
        painter.drawRect(chart_rect)

    def value_range(self, values):
        if self.fixed_range:
            return self.fixed_range
        minimum, maximum = min(values), max(values)
        if minimum == maximum:
            margin = max(1.0, abs(maximum) * 0.15)
            return minimum - margin, maximum + margin
        margin = (maximum - minimum) * 0.12
        return minimum - margin, maximum + margin

    def draw_axis_labels(self, painter, chart_rect, minimum, maximum):
        painter.setPen(QColor("#93a4bb"))
        painter.drawText(QRectF(0, chart_rect.top() - 4, 30, 16), Qt.AlignmentFlag.AlignRight, self.format_value(maximum))
        painter.drawText(QRectF(0, chart_rect.bottom() - 12, 30, 16), Qt.AlignmentFlag.AlignRight, self.format_value(minimum))

    def draw_line(self, painter, chart_rect, scene_index, values, minimum, maximum):
        color = QColor(SCENE_COLORS[scene_index])
        painter.setPen(QPen(color, 2))
        y_span = max(0.000001, maximum - minimum)
        x_span = max(1, len(values) - 1)
        points = []
        for point_index, value in enumerate(values):
            x = chart_rect.left() + chart_rect.width() * point_index / x_span
            y = chart_rect.bottom() - chart_rect.height() * (value - minimum) / y_span
            points.append(QPointF(x, y))
        for point_index in range(len(points) - 1):
            painter.drawLine(points[point_index], points[point_index + 1])

    def draw_x_axis_label(self, painter, chart_rect):
        painter.setPen(QColor("#93a4bb"))
        label = getattr(self, "x_axis_label", "Training Iterations")
        painter.drawText(QRectF(chart_rect.left(), chart_rect.bottom() + 2, chart_rect.width(), 14), Qt.AlignmentFlag.AlignCenter, label)

    def draw_legend(self, painter):
        x, y = 42, self.height() - 15
        scenes = (0,) if self.single_scene_mode else range(len(SCENE_COLORS))
        for scene_index in scenes:
            color = SCENE_COLORS[scene_index]
            painter.setPen(QPen(QColor(color), 2))
            painter.drawLine(QPointF(x, y), QPointF(x + 14, y))
            painter.setPen(QColor("#c7d2fe"))
            painter.drawText(x + 18, y + 4, f"Scene {scene_index + 1}")
            x += 68

    def format_value(self, value):
        if self.suffix:
            return f"{value:.0f}{self.suffix}"
        return f"{value:.1f}"


class AlgorithmSuccessComparisonWidget(QWidget):
    MAX_POINTS = 160
    ALGORITHMS = ("PPO", "SAC", "TD3", "DDPG")

    def __init__(self):
        super().__init__()
        self.algorithm_values = {algorithm: deque(maxlen=self.MAX_POINTS) for algorithm in self.ALGORITHMS}
        self.x_axis_label = "Training Iterations"
        self.setMinimumHeight(100)

    def set_x_axis_label(self, text: str):
        self.x_axis_label = str(text) if text else "Training Iterations"
        self.update()

    def append_algorithm_success_rate(self, algorithm_name, success_rate):
        if algorithm_name not in self.algorithm_values:
            return
        self.algorithm_values[algorithm_name].append(self.normalize_success_rate(success_rate))
        self.update()

    def set_algorithm_success_rate(self, algorithm_name, success_rate):
        if algorithm_name not in self.algorithm_values:
            return
        self.algorithm_values[algorithm_name] = deque([self.normalize_success_rate(success_rate)], maxlen=self.MAX_POINTS)
        self.update()

    def set_algorithm_success_rates(self, values):
        for algorithm_name, success_rate in values.items():
            if isinstance(success_rate, (list, tuple, deque)):
                self.set_algorithm_success_rate_values(algorithm_name, success_rate)
            else:
                self.set_algorithm_success_rate(algorithm_name, success_rate)

    def set_algorithm_success_rate_values(self, algorithm_name, values):
        if algorithm_name not in self.algorithm_values:
            return
        self.algorithm_values[algorithm_name] = deque(
            [self.normalize_success_rate(value) for value in list(values)[-self.MAX_POINTS:]],
            maxlen=self.MAX_POINTS,
        )
        self.update()

    def normalize_success_rate(self, success_rate):
        value = float(success_rate)
        if 0.0 <= value <= 1.0:
            value *= 100.0
        return max(0.0, min(100.0, value))

    def paintEvent(self, event):  # (same as before, kept for completeness)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(2, 6, 23, 44))
        chart_rect = QRectF(34.0, 14.0, max(1.0, self.width() - 48.0), max(1.0, self.height() - 46.0))
        self.draw_grid(painter, chart_rect)
        all_values = [value for values in self.algorithm_values.values() for value in values]
        if not all_values:
            painter.setPen(QColor("#64748b"))
            painter.drawText(
                chart_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Waiting for algorithm success-rate data",
            )
            self.draw_x_axis_label(painter, chart_rect)
            self.draw_legend(painter)
            return
        for index, (algorithm_name, values) in enumerate(self.algorithm_values.items()):
            if len(values) < 2:
                continue
            self.draw_line(painter, chart_rect, index, list(values))
        self.draw_x_axis_label(painter, chart_rect)
        self.draw_legend(painter)

    def draw_grid(self, painter, chart_rect):
        painter.setPen(QPen(QColor(59, 130, 246, 46), 1, Qt.PenStyle.DotLine))
        for step in range(5):
            y = chart_rect.top() + chart_rect.height() * step / 4
            painter.drawLine(QPointF(chart_rect.left(), y), QPointF(chart_rect.right(), y))
        for step in range(6):
            x = chart_rect.left() + chart_rect.width() * step / 5
            painter.drawLine(QPointF(x, chart_rect.top()), QPointF(x, chart_rect.bottom()))
        painter.setPen(QColor("#93a4bb"))
        painter.drawText(QRectF(0, chart_rect.top() - 4, 30, 16), Qt.AlignmentFlag.AlignRight, "100%")
        painter.drawText(QRectF(0, chart_rect.bottom() - 12, 30, 16), Qt.AlignmentFlag.AlignRight, "0%")
        painter.setPen(QPen(QColor(96, 165, 250, 78), 1))
        painter.drawRect(chart_rect)

    def draw_line(self, painter, chart_rect, algorithm_index, values):
        painter.setPen(QPen(QColor(ALGORITHM_COLORS[algorithm_index]), 2))
        x_span = max(1, len(values) - 1)
        points = []
        for point_index, value in enumerate(values):
            x = chart_rect.left() + chart_rect.width() * point_index / x_span
            y = chart_rect.bottom() - chart_rect.height() * value / 100.0
            points.append(QPointF(x, y))
        for point_index in range(len(points) - 1):
            painter.drawLine(points[point_index], points[point_index + 1])

    def draw_x_axis_label(self, painter, chart_rect):
        painter.setPen(QColor("#93a4bb"))
        label = getattr(self, "x_axis_label", "Training Iterations")
        painter.drawText(QRectF(chart_rect.left(), chart_rect.bottom() + 2, chart_rect.width(), 14), Qt.AlignmentFlag.AlignCenter, label)

    def draw_legend(self, painter):
        x, y = 42, self.height() - 15
        for index, algorithm_name in enumerate(self.ALGORITHMS):
            painter.setPen(QPen(QColor(ALGORITHM_COLORS[index]), 2))
            painter.drawLine(QPointF(x, y), QPointF(x + 14, y))
            painter.setPen(QColor("#c7d2fe"))
            painter.drawText(x + 18, y + 4, algorithm_name)
            x += 64


class HardwareSampler:
    def __init__(self):
        self.previous_cpu_times = self.read_cpu_times()
        self.nvidia_smi = shutil.which("nvidia-smi")
        self.cached_gpu_name = None

    def sample(self):
        gpu_info = self.read_gpu_status()
        if gpu_info.get("name"):
            self.cached_gpu_name = gpu_info["name"]
        elif not self.cached_gpu_name:
            self.cached_gpu_name = self.read_windows_gpu_name()
        return {
            "gpu":       {"text": self.cached_gpu_name or "N/A"},
            "gpu_util":  gpu_info.get("util"),
            "vram":      gpu_info.get("vram"),
            "gpu_temp":  gpu_info.get("temp"),
            "gpu_power": gpu_info.get("power"),
            "memory":    self.read_memory_usage(),
            "cpu":       self.read_cpu_usage(),
        }

    def read_cpu_usage(self):
        current = self.read_cpu_times()
        previous, self.previous_cpu_times = self.previous_cpu_times, current
        if not previous or not current:
            return None
        idle_delta, total_delta = current[0] - previous[0], current[1] - previous[1]
        if total_delta <= 0:
            return None
        percent = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))
        return {"percent": percent, "text": f"{percent:.0f}%"}

    def read_cpu_times(self):
        if not hasattr(ctypes, "windll"):
            return None
        class FileTime(ctypes.Structure):
            _fields_ = (("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong))
        idle_time, kernel_time, user_time = FileTime(), FileTime(), FileTime()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle_time), ctypes.byref(kernel_time), ctypes.byref(user_time)):
            return None
        idle = (idle_time.dwHighDateTime << 32) + idle_time.dwLowDateTime
        kernel = (kernel_time.dwHighDateTime << 32) + kernel_time.dwLowDateTime
        user = (user_time.dwHighDateTime << 32) + user_time.dwLowDateTime
        return idle, kernel + user

    def read_memory_usage(self):
        if not hasattr(ctypes, "windll"):
            return None
        class MemoryStatus(ctypes.Structure):
            _fields_ = (
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            )
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        used_gb = self.bytes_to_gb(status.ullTotalPhys - status.ullAvailPhys)
        total_gb = self.bytes_to_gb(status.ullTotalPhys)
        percent = used_gb / total_gb * 100.0 if total_gb > 0 else float(status.dwMemoryLoad)
        return {"percent": percent, "text": f"{used_gb:.1f} / {total_gb:.1f} GB ({percent:.0f}%)"}

    def read_gpu_status(self):
        """一次 nvidia-smi 调用拉取 name/util/vram/temp/power(若不可用单字段返回 None)。"""
        if not self.nvidia_smi:
            return {}
        try:
            result = subprocess.run(
                [self.nvidia_smi,
                 "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                 "temperature.gpu,power.draw,power.limit",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        first_line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        # name 可能含逗号,从末尾倒数取数字字段;最后 6 个一定是数字
        if len(parts) < 7:
            return {"name": parts[0] if parts else None}
        name = ",".join(parts[:-6]).strip()
        out = {"name": name}
        def _f(s):
            try: return float(s)
            except (ValueError, TypeError): return None
        util = _f(parts[-6])
        mem_used = _f(parts[-5])
        mem_total = _f(parts[-4])
        temp = _f(parts[-3])
        power_draw = _f(parts[-2])
        power_limit = _f(parts[-1])
        if util is not None:
            out["util"] = {"percent": util, "text": f"{util:.0f}%"}
        if mem_used is not None and mem_total is not None and mem_total > 0:
            p = mem_used / mem_total * 100.0
            out["vram"] = {
                "percent": p,
                "text": f"{self.mib_to_gb(mem_used):.1f} / {self.mib_to_gb(mem_total):.1f} GB ({p:.0f}%)",
            }
        if temp is not None:
            out["temp"] = {"text": f"{temp:.0f} °C"}
        if power_draw is not None:
            ptext = (f"{power_draw:.0f} / {power_limit:.0f} W"
                     if power_limit else f"{power_draw:.0f} W")
            out["power"] = {"text": ptext}
        return out

    def read_windows_gpu_name(self):
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name)"],
                capture_output=True, text=True, timeout=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip().splitlines()[0] if result.stdout.strip() else None

    def bytes_to_gb(self, value):
        return value / 1024 / 1024 / 1024

    def mib_to_gb(self, value):
        return value / 1024


class HardwareStatusWidget(QWidget):
    # (key, label, has_bar) — has_bar=False 时该行只有 label + value,不画进度条
    METRICS = (
            ("gpu",       "Graphics Card", False),
            ("gpu_util",  "GPU Usage",     True),
            ("vram",      "VRAM",          True),
            ("gpu_temp",  "GPU Temp.",     False),
            ("gpu_power", "GPU Power",     False),
            ("memory",    "Memory",        True),
        ("cpu",       "CPU",       True),
    )

    def __init__(self):
        super().__init__()
        self.value_labels, self.bars = {}, {}
        self.build_ui()

    def build_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(5)
        for row, (key, label_text, has_bar) in enumerate(self.METRICS):
            label = QLabel(label_text)
            label.setObjectName("HardwareMetricLabel")
            value_label = QLabel("--")
            value_label.setObjectName("HardwareMetricValue")
            if has_bar:
                value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            else:
                value_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(label, row, 0)
            if has_bar:
                bar = QProgressBar()
                bar.setObjectName("HardwareMetricBar")
                bar.setRange(0, 100)
                bar.setTextVisible(False)
                bar.setFixedHeight(8)
                layout.addWidget(bar, row, 1)
                layout.addWidget(value_label, row, 2)
                self.bars[key] = bar
            else:
                layout.addWidget(value_label, row, 1, 1, 2)
            self.value_labels[key] = value_label
            layout.setRowStretch(row, 0)
        layout.setColumnStretch(1, 1)
        # 软件版本静态行 — 训练复现性用
        version_row = len(self.METRICS)
        version_label = QLabel(self._build_version_text())
        version_label.setObjectName("HardwareMetricVersionLabel")
        version_label.setStyleSheet(
            "color: #93a4bb; font-size: 10px; padding-top: 6px;"
            "border-top: 1px solid rgba(96, 165, 250, 0.25); margin-top: 4px;"
        )
        version_label.setWordWrap(True)
        layout.addWidget(version_label, version_row, 0, 1, 3)
        layout.setRowStretch(version_row, 0)
        # 末尾 stretch 吸高
        layout.setRowStretch(version_row + 1, 1)

    @staticmethod
    def _build_version_text() -> str:
        import sys
        parts = [f"Python {sys.version_info.major}.{sys.version_info.minor}."
                 f"{sys.version_info.micro}"]
        try:
            import torch  # type: ignore
            parts.append(f"PyTorch {torch.__version__.split('+')[0]}")
            if torch.cuda.is_available():
                parts.append(f"CUDA {torch.version.cuda}")
        except Exception:
            pass
        try:
            from zmd.simulation.mujoco_loader import import_mujoco
            mujoco = import_mujoco()
            parts.append(f"MuJoCo {mujoco.__version__}")
        except Exception:
            pass
        try:
            import stable_baselines3  # type: ignore
            parts.append(f"sb3 {stable_baselines3.__version__}")
        except Exception:
            pass
        return " · ".join(parts)

    def update_values(self, values):
        for key, _, has_bar in self.METRICS:
            value = values.get(key)
            if value is None:
                if key in self.bars:
                    self.bars[key].setValue(0)
                self.value_labels[key].setText("N/A")
                continue
            if has_bar:
                percent = max(0, min(100, int(round(value.get("percent", 0)))))
                self.bars[key].setValue(percent)
                self.value_labels[key].setText(value.get("text", f"{percent}%"))
            else:
                self.value_labels[key].setText(value.get("text", "N/A"))


class WorkbenchDebugPanel(QFrame):
    MAX_LINES = 400

    def __init__(self):
        super().__init__()
        self.messages = []
        self.hardware_sampler = HardwareSampler()
        self.setObjectName("DebugPanel")
        self.setFixedHeight(340)
        self.setStyleSheet(DEBUG_PANEL_STYLE)
        self.build_ui()
        self.hardware_timer = QTimer(self)
        self.hardware_timer.timeout.connect(self.refresh_hardware_usage)
        self.hardware_timer.start(2000)
        self.refresh_hardware_usage()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel("Debug Info")
        title_label.setObjectName("DebugPanelTitle")
        title_row.addWidget(title_label)

        # 主视图切换:调试/硬件 (左右平分) ↔ 4 条训练曲线
        self.mode_debug_button = self.make_slot_switch_button("Debug + Hardware", True)
        self.mode_curves_button = self.make_slot_switch_button("Training Curves", False)
        self.mode_debug_button.clicked.connect(lambda: self.set_view_mode(0))
        self.mode_curves_button.clicked.connect(lambda: self.set_view_mode(1))
        title_row.addSpacing(8)
        title_row.addWidget(self.mode_debug_button)
        title_row.addWidget(self.mode_curves_button)

        title_row.addStretch(1)
        # 训练状态行 — 由 workspace.set_training_status() 喂数据;空时不显示
        self.training_status_label = QLabel("")
        self.training_status_label.setStyleSheet(
            "color: #93c5fd; font-size: 11px; font-weight: 700; "
            "padding: 2px 8px; background: rgba(30, 64, 175, 0.30); "
            "border: 1px solid rgba(96, 165, 250, 0.45); border-radius: 6px;"
        )
        self.training_status_label.setVisible(False)
        title_row.addWidget(self.training_status_label)
        # 清空日志按钮(任何模式下都可点,只清调试日志)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("DebugPanelClearButton")
        self.clear_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_button.clicked.connect(self.clear)
        title_row.addWidget(self.clear_button)

        # ---- 调试日志 + 硬件状态(各自一块,左右平分)----
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("DebugPanelLog")
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.hardware_status = HardwareStatusWidget()

        self.section_debug = DebugSection("Debug", self.log_view)
        self.section_hw = DebugSection("Hardware", self.hardware_status)
        page_debug_hw = QWidget()
        dh_layout = QHBoxLayout(page_debug_hw)
        dh_layout.setContentsMargins(0, 0, 0, 0)
        dh_layout.setSpacing(8)
        dh_layout.addWidget(self.section_debug, 1)
        dh_layout.addWidget(self.section_hw, 1)

        # ---- 4 条论文级曲线 — 1×4 横排 ----
        self.reward_curve = SceneCurveWidget("Waiting for reward data")
        self.success_curve = SceneCurveWidget("Waiting for success-rate data",
                                              fixed_range=(0.0, 100.0), suffix="%")
        self.length_curve = SceneCurveWidget("Waiting for episode-length data")
        self.blocked_curve = SceneCurveWidget("Waiting for blocked-count data")
        self.section_reward = DebugSection("Evaluation Mean Return", self.reward_curve)
        self.section_success = DebugSection("Evaluation Success Rate", self.success_curve)
        self.section_length = DebugSection("Evaluation Mean Length", self.length_curve)
        self.section_blocked = DebugSection("Evaluation Mean Blocked Count", self.blocked_curve)
        page_curves = QWidget()
        cv_layout = QHBoxLayout(page_curves)
        cv_layout.setContentsMargins(0, 0, 0, 0)
        cv_layout.setSpacing(8)
        cv_layout.addWidget(self.section_reward, 1)
        cv_layout.addWidget(self.section_success, 1)
        cv_layout.addWidget(self.section_length, 1)
        cv_layout.addWidget(self.section_blocked, 1)

        # ---- 主视图 stacked(整面板只显示其一)----
        self.main_view_stack = QStackedWidget()
        self.main_view_stack.addWidget(page_debug_hw)   # idx 0
        self.main_view_stack.addWidget(page_curves)     # idx 1
        self.main_view_stack.setCurrentIndex(0)

        layout.addLayout(title_row)
        layout.addWidget(self.main_view_stack, 1)

    def set_view_mode(self, index: int):
        """0 = 调试+硬件,1 = 训练曲线;整个面板只显示其一。"""
        index = 0 if index == 0 else 1
        self.main_view_stack.setCurrentIndex(index)
        self.mode_debug_button.setChecked(index == 0)
        self.mode_curves_button.setChecked(index == 1)
        # 清空按钮只在调试模式下有意义,但保留全局可点(不会清曲线)

    def set_training_status(self, text: str):
        """显示当前训练统计(ep/r/avg/succ)。传空串会隐藏。"""
        text = (text or "").strip()
        self.training_status_label.setText(text)
        self.training_status_label.setVisible(bool(text))

    def set_curves_x_axis_label(self, label: str):
        """同步训练曲线 X 轴标签(用 'ep=N/total' 把训练次数挂上来)。"""
        for w in (self.reward_curve, self.success_curve,
                  self.length_curve, self.blocked_curve):
            if hasattr(w, "set_x_axis_label"):
                w.set_x_axis_label(label)

    def append_episode_length(self, scene_index: int, length: int):
        self.length_curve.append_value(scene_index, float(length))

    def append_blocked_count(self, scene_index: int, count: int):
        self.blocked_curve.append_value(scene_index, float(count))

    def set_single_scene_mode(self, single: bool):
        """切换单场景 / 多场景:改 section 标题 + 控制曲线 widget 模式。"""
        if single:
            self.section_reward.set_title("Evaluation Mean Return")
            self.section_success.set_title("Evaluation Success Rate")
            self.section_length.set_title("Evaluation Mean Length")
            self.section_blocked.set_title("Evaluation Mean Blocked Count")
        else:
            self.section_reward.set_title("4-Scene Evaluation Mean Return")
            self.section_success.set_title("4-Scene Evaluation Success Rate")
            self.section_length.set_title("4-Scene Evaluation Mean Length")
            self.section_blocked.set_title("4-Scene Evaluation Mean Blocked Count")
        for w in (self.reward_curve, self.success_curve,
                  self.length_curve, self.blocked_curve):
            if hasattr(w, "set_single_scene_mode"):
                w.set_single_scene_mode(single)

    def make_slot_switch_button(self, text, checked):
        button = QPushButton(text)
        button.setObjectName("DebugSlotSwitchButton")
        button.setCheckable(True)
        button.setChecked(checked)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def append_message(self, message, scene_index=None):
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        prefix = f"Experiment Scene {scene_index + 1} | " if scene_index is not None else ""
        self.messages.append(f"[{timestamp}] {prefix}{message}")
        if len(self.messages) > self.MAX_LINES:
            self.messages = self.messages[-self.MAX_LINES:]
        self.log_view.setPlainText("\n".join(self.messages))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def append_reward(self, scene_index, reward):
        self.reward_curve.append_value(scene_index, reward)

    def append_success_rate(self, scene_index, success_rate):
        value = float(success_rate)
        if 0.0 <= value <= 1.0:
            value *= 100.0
        self.success_curve.append_value(scene_index, value)

    def set_reward_values(self, scene_index, values):
        self.reward_curve.set_scene_values(scene_index, values)

    def set_success_rate_values(self, scene_index, values):
        normalized_values = [(float(value) * 100.0 if 0.0 <= float(value) <= 1.0 else float(value)) for value in values]
        self.success_curve.set_scene_values(scene_index, normalized_values)

    def refresh_hardware_usage(self):
        self.hardware_status.update_values(self.hardware_sampler.sample())

    def clear(self):
        self.messages.clear()
        self.log_view.clear()
        self.reward_curve.clear_values()
        self.success_curve.clear_values()
        self.length_curve.clear_values()
        self.blocked_curve.clear_values()
