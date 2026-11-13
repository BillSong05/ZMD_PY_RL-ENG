from pathlib import Path

import pyvista as pv
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor


# 职责：封装 PyVista/VTK 三维视窗，提供多模型加载、标记点渲染和相机预设接口。
class Scene3DWidget(QWidget):
    status_changed = pyqtSignal(str)
    # workspace 监听 — 3D 视窗里按 WSADQEZ 时转发到主窗口
    key_forwarded = pyqtSignal(int, bool)              # (Qt key, pressed)

    DEFAULT_NORMALIZED_LENGTH = 900.0
    DEFAULT_MODEL_SCALE = 0.1
    DEFAULT_MARKER_RADIUS = 2.6

    CAMERA_PRESETS = {
        "isometric": [(120.0, -140.0, 105.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        "front": [(0.0, -160.0, 35.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        "left": [(-160.0, 0.0, 35.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        "top": [(0.0, 0.0, 180.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    }

    def __init__(self, camera_preset="isometric", parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        self.setStyleSheet("background: #07101f;")
        self.camera_preset = camera_preset
        self.models = {}
        self.markers = {}
        self.marker_radius = self.DEFAULT_MARKER_RADIUS
        self._shutting_down = False
        self.plotter = QtInteractor(self)
        self.build_ui()
        self.configure_plotter()
        self.build_world()
        self.set_camera_preset(self.camera_preset)

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plotter.interactor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.plotter.interactor, 1)
        # 在 Qt 层拦截 QtInteractor 收到的 ↑↓←→，VTK 自然收不到
        try:
            self.plotter.interactor.installEventFilter(self)
        except Exception:
            pass
        # 双保险：VTK observer 端也清空 KeySym
        try:
            iren = self.plotter.iren
            iren.AddObserver("KeyPressEvent", self._suppress_arrow_keys, 10.0)
        except Exception:
            pass

    _ARROW_KEYS = {
        Qt.Key.Key_Up, Qt.Key.Key_Down,
        Qt.Key.Key_Left, Qt.Key.Key_Right,
        Qt.Key.Key_PageUp, Qt.Key.Key_PageDown,
    }
    _FORWARD_KEYS = {
        Qt.Key.Key_W, Qt.Key.Key_S, Qt.Key.Key_A, Qt.Key.Key_D,
        Qt.Key.Key_Q, Qt.Key.Key_E, Qt.Key.Key_Z,
    }

    def eventFilter(self, watched, event):
        """Qt 层拦截：方向键消化 + WSADQEZ 转发到 workspace。

        关键：所有 W/S/A/D/Q/E/Z 必须**始终**被消化（return True），不能因为
        isAutoRepeat 等条件而漏过 — 漏过会让 VTK 接到 W → 把 STL 切成 wireframe。
        forward 信号只在第一次按下/释放发，避免重复触发逻辑。
        """
        try:
            if watched is self.plotter.interactor and event.type() in (
                QEvent.Type.KeyPress, QEvent.Type.KeyRelease,
                QEvent.Type.ShortcutOverride,
            ):
                k = event.key()
                if k in self._ARROW_KEYS:
                    event.accept()
                    return True
                if k in self._FORWARD_KEYS:
                    # 仅首次（非 autorepeat）发 forward 信号；其余事件依然吞掉
                    if not event.isAutoRepeat():
                        if event.type() == QEvent.Type.KeyPress:
                            self.key_forwarded.emit(int(k), True)
                        elif event.type() == QEvent.Type.KeyRelease:
                            self.key_forwarded.emit(int(k), False)
                    event.accept()
                    return True
        except Exception:
            pass
        return super().eventFilter(watched, event)

    def configure_plotter(self):
        # 深灰黑背景 — 介入/内窥镜风格，不纯黑也不太亮
        self.plotter.set_background("#0d1116", top="#161b22")
        self.plotter.enable_anti_aliasing()
        # 柔和相机灯：低强度避免过曝
        try:
            head_light = pv.Light(light_type="cameralight", intensity=0.35)
            head_light.cone_angle = 70
            self.plotter.add_light(head_light)
            self._head_light = head_light
        except Exception:
            self._head_light = None
        # 弱环境/补光，避免内部纯黑
        try:
            ambient = pv.Light(light_type="scene light", intensity=0.18,
                               position=(50, -50, 30), focal_point=(0, 0, 0))
            self.plotter.add_light(ambient)
        except Exception:
            pass

    # VTK 默认 W=wireframe / S=surface — 我们用 WSAD/QE 当 debug ball 方向键
    # Z 用作 ground grid 显隐 toggle — 都不能让 VTK InteractorStyle 抢去
    _SUPPRESSED_VTK_KEYS = {
        "Up", "Down", "Left", "Right", "Prior", "Next",
        "w", "W", "s", "S", "a", "A", "d", "D",
        "q", "Q", "e", "E", "z", "Z",
    }

    def _suppress_arrow_keys(self, iren, _event):
        """清空 KeySym，避免 VTK 把 ↑↓←→ 和 WSADQEZ 转成 dolly / wireframe 等。"""
        try:
            key = iren.GetKeySym() or ""
            if key in self._SUPPRESSED_VTK_KEYS:
                iren.SetKeySym("")
        except Exception:
            pass

    def set_ground_grid_visible(self, visible: bool) -> None:
        actor = getattr(self, "_ground_grid_actor", None)
        if actor is None:
            return
        try:
            actor.SetVisibility(bool(visible))
            self._ground_grid_visible = bool(visible)
            self.plotter.render()
        except Exception:
            pass

    def toggle_ground_grid(self) -> bool:
        cur = bool(getattr(self, "_ground_grid_visible", True))
        self.set_ground_grid_visible(not cur)
        return not cur

    def set_followlight_intensity(self, intensity: float) -> None:
        """跟随模式可以调暗 headlight，避免 lumen 过曝。"""
        if self._head_light is not None:
            try:
                self._head_light.intensity = float(intensity)
                self.plotter.render()
            except Exception:
                pass

    def build_world(self):
        self.add_ground_grid()
        self.plotter.add_axes(color="white")

    def add_ground_grid(self, size=120.0, spacing=10.0, z_level=-12.0):
        lines = []
        start = -size / 2.0
        end = size / 2.0
        count = int(size / spacing) + 1
        for index in range(count):
            coordinate = start + index * spacing
            lines.append(pv.Line((start, coordinate, z_level), (end, coordinate, z_level)))
            lines.append(pv.Line((coordinate, start, z_level), (coordinate, end, z_level)))
        grid = lines[0]
        for line in lines[1:]:
            grid = grid.merge(line)
        self._ground_grid_actor = self.plotter.add_mesh(
            grid, color="#335170", line_width=1, opacity=0.45,
            name="ground_grid", pickable=False)
        self._ground_grid_visible = True

    def clear_models(self):
        for model in self.models.values():
            self.plotter.remove_actor(model["actor"], render=False)
        for marker in self.markers.values():
            self.plotter.remove_actor(marker, render=False)
        self.models.clear()
        self.markers.clear()
        self.plotter.render()

    def load_model(self, model_path, model_key=None, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0),
                   scale=DEFAULT_MODEL_SCALE, color="#dbeafe", opacity=1.0, reset_camera=True,
                   mesh=None, specular=0.35, ambient=None, diffuse=None, specular_power=20):
        path = Path(model_path)
        if not path.exists() and mesh is None:
            self.status_changed.emit(f"Model file not found: {path}")
            return None
        key = model_key or path.stem
        try:
            if mesh is None:
                mesh = self.prepare_mesh(path)
            if key in self.models:
                self.plotter.remove_actor(self.models[key]["actor"], render=False)
            kw = dict(color=color, specular=specular, specular_power=specular_power,
                      smooth_shading=True,
                      opacity=max(0.0, min(1.0, float(opacity))))
            if ambient is not None: kw["ambient"] = float(ambient)
            if diffuse is not None: kw["diffuse"] = float(diffuse)
            actor = self.plotter.add_mesh(mesh, **kw)
            model_data = {
                "actor": actor, "path": path, "mesh": mesh,
                "opacity": max(0.0, min(1.0, float(opacity))),
                "transform": {"position": tuple(position), "rotation": tuple(rotation), "scale": float(scale)},
            }
            self.models[key] = model_data
            self.apply_actor_transform(key, render=False)
            if reset_camera:
                self.reset_view()
            else:
                self.plotter.render()
            self.status_changed.emit(f"Model loaded: {path.name}")
            return actor
        except Exception as error:
            self.status_changed.emit(f"Failed to load model: {path.name}, {error}")
            return None

    def prepare_mesh(self, path):
        mesh = pv.read(path).copy()
        mesh.translate((-mesh.center[0], -mesh.center[1], -mesh.center[2]), inplace=True)
        if mesh.length > 0:
            mesh.scale(self.DEFAULT_NORMALIZED_LENGTH / mesh.length, inplace=True)
        return mesh

    def set_model_transform(self, model_key, position=None, rotation=None, scale=None):
        if model_key not in self.models:
            return
        transform = self.models[model_key]["transform"]
        if position is not None:
            transform["position"] = tuple(position)
        if rotation is not None:
            transform["rotation"] = tuple(rotation)
        if scale is not None:
            transform["scale"] = float(scale)
        self.apply_actor_transform(model_key, render=True)

    def get_model_transform(self, model_key):
        if model_key not in self.models:
            return None
        transform = self.models[model_key]["transform"]
        return {"position": tuple(transform["position"]), "rotation": tuple(transform["rotation"]), "scale": float(transform["scale"])}

    def set_model_opacity(self, model_key, opacity):
        if model_key not in self.models:
            return
        bounded_opacity = max(0.0, min(1.0, float(opacity)))
        self.models[model_key]["opacity"] = bounded_opacity
        self.models[model_key]["actor"].GetProperty().SetOpacity(bounded_opacity)
        self.plotter.render()

    def set_markers(self, entry_point, exit_point, marker_radius=None):
        if marker_radius is not None:
            self.set_marker_radius(marker_radius, render=False)
        self.set_marker("entry", entry_point, "#22c55e")
        self.set_marker("exit", exit_point, "#f97316")
        self.plotter.render()

    def set_marker_radius(self, marker_radius, render=True):
        self.marker_radius = max(0.1, float(marker_radius))
        for marker in self.markers.values():
            marker.SetScale(self.marker_radius, self.marker_radius, self.marker_radius)
        if render:
            self.plotter.render()

    def set_marker(self, marker_key, position, color):
        if marker_key not in self.markers:
            marker_mesh = pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=24)
            self.markers[marker_key] = self.plotter.add_mesh(
                marker_mesh, color=color, specular=0.4, specular_power=24, pickable=False, render=False,
            )
        self.markers[marker_key].SetScale(self.marker_radius, self.marker_radius, self.marker_radius)
        self.markers[marker_key].SetPosition(*position)

    def apply_actor_transform(self, model_key, render):
        model = self.models[model_key]
        actor = model["actor"]
        transform = model["transform"]
        actor.SetPosition(*transform["position"])
        actor.SetOrientation(*transform["rotation"])
        actor.SetScale(transform["scale"], transform["scale"], transform["scale"])
        if render:
            self.plotter.render()

    def set_camera(self, camera_position):
        self.plotter.camera_position = camera_position
        self.plotter.render()

    def set_camera_preset(self, preset):
        self.camera_preset = preset
        preset_position, preset_focal, preset_up = self.CAMERA_PRESETS.get(preset, self.CAMERA_PRESETS["isometric"])
        self.set_camera([preset_position, preset_focal, preset_up])

    def remove_actor(self, actor) -> None:
        """从 plotter 中移除指定 actor 并刷新。"""
        self.plotter.remove_actor(actor)
        self.plotter.render()

    def clear_scene(self) -> None:
        """清除场景中所有模型和标记。"""
        self.clear_models()

    def reset_view(self) -> None:
        self.reset_camera()

    def reset_camera(self) -> None:
        """重置相机到默认视角。"""
        self.plotter.reset_camera()
        self.set_camera_preset(self.camera_preset)

    def shutdown(self) -> None:
        """Release the VTK/OpenGL window before Qt destroys the native widget."""
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            self.plotter.interactor.removeEventFilter(self)
        except Exception:
            pass
        try:
            self.plotter.iren.Disable()
        except Exception:
            pass
        try:
            self.plotter.clear()
        except Exception:
            pass
        try:
            self.plotter.close()
        except Exception:
            pass
        try:
            self.plotter.interactor.close()
        except Exception:
            pass

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def focus_camera_on(self, focal_point, keep_direction: bool = True) -> None:
        """让相机看向 focal_point；keep_direction=True 时保持原相机朝向。"""
        camera = self.plotter.camera
        old_position = list(camera.position)
        old_focal = list(camera.focal_point)
        offset = [old_position[i] - old_focal[i] for i in range(3)]
        new_focal = [float(v) for v in focal_point]
        if keep_direction:
            new_position = [new_focal[i] + offset[i] for i in range(3)]
        else:
            new_position = old_position
        camera.position = tuple(new_position)
        camera.focal_point = tuple(new_focal)
        self.plotter.render()

    def set_first_person_view(self, camera_position, tip, tangent,
                              forward_distance: float = 8.0) -> None:
        import math
        # ---- 有效性检查 ----
        try:
            cp = [float(camera_position[i]) for i in range(3)]
            tp = [float(tip[i]) for i in range(3)]
            tg = [float(tangent[i]) for i in range(3)]
        except Exception:
            return
        for v in cp + tp + tg:
            if not math.isfinite(v):
                return
        t_norm = math.sqrt(tg[0] * tg[0] + tg[1] * tg[1] + tg[2] * tg[2])
        if t_norm < 1e-6:
            return
        dx, dy, dz = cp[0] - tp[0], cp[1] - tp[1], cp[2] - tp[2]
        if math.sqrt(dx * dx + dy * dy + dz * dz) < 1e-6:
            return
        return self._do_set_first_person_view(cp, tp, tg, t_norm, forward_distance)

    def _do_set_first_person_view(self, cp, tp, tg, t_norm, forward_distance):
        """正后方第一人称 — 已通过外层有效性检查。"""
        import math

        camera_position = cp
        tip = tp
        tx, ty, tz = tg[0] / t_norm, tg[1] / t_norm, tg[2] / t_norm

        # up = world_up - (world_up · t) * t  （把 world up 投影到切线正交面）
        wux, wuy, wuz = 0.0, 0.0, 1.0
        dot = tx * wux + ty * wuy + tz * wuz
        ux = wux - dot * tx
        uy = wuy - dot * ty
        uz = wuz - dot * tz
        unorm = math.sqrt(ux * ux + uy * uy + uz * uz)
        if unorm < 1e-6:
            # 切线垂直于世界 up（极少见），改用 X 作辅助
            wux, wuy, wuz = 1.0, 0.0, 0.0
            dot = tx * wux + ty * wuy + tz * wuz
            ux = wux - dot * tx
            uy = wuy - dot * ty
            uz = wuz - dot * tz
            unorm = math.sqrt(ux * ux + uy * uy + uz * uz)
        if unorm < 1e-6:
            ux, uy, uz = 0.0, 0.0, 1.0
        else:
            ux, uy, uz = ux / unorm, uy / unorm, uz / unorm

        focal = (
            float(tip[0]) + tx * forward_distance,
            float(tip[1]) + ty * forward_distance,
            float(tip[2]) + tz * forward_distance,
        )
        camera = self.plotter.camera
        camera.position = (float(camera_position[0]),
                           float(camera_position[1]),
                           float(camera_position[2]))
        camera.focal_point = focal
        camera.up = (ux, uy, uz)
        # 让 VTK 按当前场景 bounds 重算合理的 near/far，避免血管被裁
        try:
            self.plotter.renderer.ResetCameraClippingRange()
        except Exception:
            camera.clipping_range = (0.05, 500.0)
        self.plotter.render()
