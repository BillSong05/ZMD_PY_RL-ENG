"""机器人模型渲染器 — 负责加载机器人 STL 到 PyVista 视窗。"""
from pathlib import Path
from typing import Optional


class RobotRenderer:
    """机器人 STL 渲染器，封装加载/显示/隐藏/清除逻辑。"""

    def __init__(self, view, mesh_cache=None):
        self.view = view
        self.actor: Optional[object] = None
        self.model_key = "robot"
        self.mesh_cache = mesh_cache

    def load(self, model_path) -> bool:
        """加载机器人 STL 模型。

        Args:
            model_path: str 或 Path，如果为 None 则跳过加载。

        Returns:
            True 如果加载成功，False 如果跳过或失败。
        """
        if model_path is None:
            return False
        path = Path(model_path)
        if not path.exists():
            self.view.status_changed.emit(f"[RobotRenderer] Model not found: {path}")
            return False
        cached_mesh = None
        if self.mesh_cache is not None:
            try:
                cached_mesh = self.mesh_cache.get_normalized(
                    path, normalized_length=self.view.DEFAULT_NORMALIZED_LENGTH,
                )
            except Exception:
                cached_mesh = None
        self.actor = self.view.load_model(
            path,
            model_key=self.model_key,
            position=(-11.2, -78.0, -16.65),
            rotation=(-90.0, 0.0, -90.0),
            scale=self.view.DEFAULT_MODEL_SCALE,
            color="#a78bfa",
            opacity=1.0,
            reset_camera=False,
            mesh=cached_mesh,
        )
        return self.actor is not None

    def show(self) -> None:
        """显示机器人模型。"""
        if self.actor is not None:
            self.actor.SetVisibility(True)
            self.view.plotter.render()

    def hide(self) -> None:
        """隐藏机器人模型。"""
        if self.actor is not None:
            self.actor.SetVisibility(False)
            self.view.plotter.render()

    def clear(self) -> None:
        """从视窗中移除机器人模型。"""
        if self.actor is not None:
            self.view.plotter.remove_actor(self.actor)
            self.actor = None
            self.view.plotter.render()
