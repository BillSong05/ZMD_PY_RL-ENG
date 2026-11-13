"""场景管理器 — 负责根据 ProjectConfig 加载模型并管理 4 个 Scene3DWidget。

支持每个场景独立的中心线 / 导丝渲染参数与相机跟随状态。
"""
from typing import Any, Dict, Optional

import numpy as np

from zmd.centerline.centerline_model import Centerline
from zmd.io.project_loader import ProjectConfig
from zmd.rendering.centerline_renderer import CenterlineRenderer
from zmd.rendering.vessel_renderer import VesselRenderer
from zmd.rendering.robot_renderer import RobotRenderer
from zmd.rendering.guidewire_renderer import GuidewireRenderer
from zmd.scene.scene_config import (
    DEFAULT_CENTERLINE_SETTINGS,
    DEFAULT_GUIDEWIRE_SETTINGS,
    DEFAULT_VASCULAR_SETTINGS,
)


class SceneManager:
    """管理多个 3D 视窗的模型加载和运行时状态刷新。"""

    def __init__(self, project_config: ProjectConfig, views: Dict[int, Any],
                 mesh_cache: Any = None):
        self.project_config = project_config
        self.views = views
        self.mesh_cache = mesh_cache
        self.centerlines: Dict[int, Centerline] = {}

        self.renderers: Dict[int, Dict[str, Any]] = {}
        self.follow_enabled: Dict[int, bool] = {}
        self.guidewire_radius: Dict[int, float] = {}
        self._scene_initialized: Dict[int, bool] = {}
        for idx, view in self.views.items():
            self.renderers[idx] = {
                "vessel": VesselRenderer(view, mesh_cache=mesh_cache),
                "robot": RobotRenderer(view, mesh_cache=mesh_cache),
                "guidewire": GuidewireRenderer(view),
                "centerline": CenterlineRenderer(view),
            }
            self.follow_enabled[idx] = False
            self.guidewire_radius[idx] = DEFAULT_GUIDEWIRE_SETTINGS["radius"]
            self._scene_initialized[idx] = False

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def ensure_initialized(self, scene_index: int) -> bool:
        """首次访问场景时加载 vessel / robot / markers / camera。已初始化则跳过。"""
        if self._scene_initialized.get(scene_index):
            return True
        if scene_index not in self.views:
            return False
        self._init_single_scene(scene_index)
        self._scene_initialized[scene_index] = True
        return True

    def load_project(self, scene_index: int = None) -> None:
        """无参数：兼容旧调用，初始化全部场景。
        指定 scene_index：仅初始化该场景（用于 lazy load）。"""
        if scene_index is not None:
            self.ensure_initialized(scene_index)
            return
        for idx in self.views:
            self.ensure_initialized(idx)

    def _init_single_scene(self, idx: int) -> None:
        vessel_path = self.project_config.vessel_model_path
        if vessel_path.exists():
            self.renderers[idx]["vessel"].load(vessel_path)
            self.renderers[idx]["vessel"].set_opacity(DEFAULT_VASCULAR_SETTINGS["opacity"])
        else:
                self.views[idx].status_changed.emit(f"Vessel model not found: {vessel_path}")
        robot_path = self.project_config.robot_model_path
        if robot_path is not None:
            if robot_path.exists():
                self.renderers[idx]["robot"].load(robot_path)
            else:
                self.views[idx].status_changed.emit(f"Robot model not found: {robot_path}")
        # markers + camera
        settings = DEFAULT_VASCULAR_SETTINGS
        self.views[idx].set_markers(settings["entry_point"], settings["exit_point"],
                                    settings["marker_radius"])
        self.views[idx].reset_camera()

    def load_vessel(self) -> None:
        """旧 API — 兼容用，全部场景加载。"""
        for idx in self.views:
            self.ensure_initialized(idx)

    def load_robot(self) -> None:
        """旧 API — 通过 ensure_initialized 间接处理。"""
        for idx in self.views:
            self.ensure_initialized(idx)

    def load_centerline(self, centerline: Centerline, scene_index: Optional[int] = None) -> None:
        """在指定场景（或全部场景）渲染中心线。"""
        targets = [scene_index] if scene_index is not None else list(self.views.keys())
        for idx in targets:
            self.centerlines[idx] = centerline
            renderer = self.renderers[idx]["centerline"]
            renderer.load(centerline)
            self.views[idx].status_changed.emit(
                f"Centerline loaded: {centerline.n_points} points, "
                f"radius {'available' if centerline.radius is not None else 'unavailable'}"
            )

    def _setup_markers(self) -> None:
        """旧 API — 现在 markers 在 _init_single_scene 中设置。"""
        settings = DEFAULT_VASCULAR_SETTINGS
        for idx, view in self.views.items():
            if self._scene_initialized.get(idx):
                view.set_markers(settings["entry_point"], settings["exit_point"],
                                 settings["marker_radius"])

    def set_markers(self, scene_index: Optional[int], entry_point, exit_point,
                    marker_radius: Optional[float] = None) -> None:
        targets = [scene_index] if scene_index is not None else list(self.views.keys())
        for idx in targets:
            self.views[idx].set_markers(entry_point, exit_point, marker_radius)

    # ------------------------------------------------------------------
    # 中心线参数
    # ------------------------------------------------------------------

    def apply_centerline_settings(self, scene_index: Optional[int],
                                  radius: float, visible: bool,
                                  show_all_branches: bool) -> None:
        targets = [scene_index] if scene_index is not None else list(self.views.keys())
        for idx in targets:
            r = self.renderers[idx]["centerline"]
            r.set_radius(radius)
            r.set_visible(visible)
            r.set_show_all_branches(show_all_branches)

    # ------------------------------------------------------------------
    # 导丝参数
    # ------------------------------------------------------------------

    def apply_guidewire_radius(self, scene_index: Optional[int], radius: float) -> None:
        targets = [scene_index] if scene_index is not None else list(self.views.keys())
        for idx in targets:
            self.guidewire_radius[idx] = float(radius)
            self.renderers[idx]["guidewire"].set_radius(radius)

    def apply_guidewire_color(self, scene_index: Optional[int], color: str) -> None:
        targets = [scene_index] if scene_index is not None else list(self.views.keys())
        for idx in targets:
            self.renderers[idx]["guidewire"].set_color(color)

    def apply_guidewire_style(self, scene_index: Optional[int], style_name: str) -> None:
        """切换某场景的导丝渲染样式。会清空旧 actor。"""
        targets = [scene_index] if scene_index is not None else list(self.views.keys())
        for idx in targets:
            r = self.renderers[idx]["guidewire"]
            r.set_style(style_name)
            self.guidewire_radius[idx] = float(r.radius)

    # ------------------------------------------------------------------
    # 相机跟随
    # ------------------------------------------------------------------

    def set_follow_enabled(self, scene_index: int, enabled: bool,
                           user_centerline_visible: bool = True,
                           user_show_all_branches: bool = True) -> None:
        """follow 状态。主场景渲染不再变化 —— 由 workspace 控制右上角 inset 显隐。"""
        del user_centerline_visible, user_show_all_branches
        self.follow_enabled[scene_index] = bool(enabled)

    def is_follow_enabled(self, scene_index: int) -> bool:
        return self.follow_enabled.get(scene_index, False)

    # ------------------------------------------------------------------
    # 相机
    # ------------------------------------------------------------------

    def reset_cameras(self) -> None:
        for view in self.views.values():
            view.reset_camera()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def toggle_vessel_visible(self, scene_index: int) -> bool:
        """切换某场景血管 STL 显隐；返回 toggle 后是否可见。"""
        r = self.renderers.get(scene_index, {}).get("vessel")
        if r is None or r.actor is None:
            return True
        try:
            visible_now = bool(r.actor.GetVisibility())
            if visible_now:
                r.hide()
            else:
                r.show()
            return not visible_now
        except Exception:
            return True

    def toggle_vessel_wireframe(self, scene_index: int) -> bool:
        """切换血管 STL 线框 / 实体表面；返回切后是否为线框。"""
        r = self.renderers.get(scene_index, {}).get("vessel")
        if r is None:
            return False
        try:
            return bool(r.toggle_wireframe())
        except Exception:
            return False

    def clear(self) -> None:
        for rgroup in self.renderers.values():
            rgroup["vessel"].clear()
            rgroup["robot"].clear()
            rgroup["guidewire"].clear()
            rgroup["centerline"].clear()

    # ------------------------------------------------------------------
    # 运行时状态刷新
    # ------------------------------------------------------------------

    # 注：旧的全屏第一人称（修改主相机）功能已移除。
    # follow 状态现在只供右上角 inset viewport 使用，由 workspace 直接处理。
    TIP_TAIL_POINTS = 30      # 跟随小窗只渲染 tip 后面 N 个点

    def update_from_state(self, state: Any, scene_index: int) -> None:
        """只刷新指定场景的 guidewire mesh 和相机。其他场景保持不变。"""
        try:
            self._update_from_state_impl(state, scene_index)
        except Exception as exc:
            # 静默捕获，保留上一帧合法相机，避免单次异常打挂场景
            try:
                self.views[scene_index].status_changed.emit(
                    f"[follow] update skipped: {type(exc).__name__}: {exc}")
            except Exception:
                pass

    def _update_from_state_impl(self, state: Any, scene_index: int) -> None:
        """只更新主场景导丝 mesh。不再修改主相机 — follow 由 inset 小窗处理。"""
        if state is None or scene_index not in self.renderers:
            return
        gw = getattr(state, "guidewire", None)
        if gw is None:
            return
        visible = getattr(gw, "centerline", None)
        if visible is None or len(visible) < 2:
            return
        visible = np.asarray(visible)
        if not np.all(np.isfinite(visible)):
            return

        gw_radius = self.guidewire_radius.get(scene_index,
                                              DEFAULT_GUIDEWIRE_SETTINGS["radius"])
        renderer = self.renderers[scene_index]["guidewire"]
        renderer.update_centerline(visible, radius=gw_radius)
