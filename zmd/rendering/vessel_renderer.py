"""血管模型渲染器 — 负责加载血管 STL 到 PyVista 视窗。"""
from pathlib import Path
from typing import Optional


class VesselRenderer:
    """血管 STL 渲染器，封装加载/显示/隐藏/清除逻辑。

    SceneManager 为每个 3D 视窗创建一个 VesselRenderer 实例，
    通过 renderer.load(path) 驱动实际的 PyVista add_mesh。
    """

    def __init__(self, view, mesh_cache=None):
        self.view = view
        self.actor: Optional[object] = None
        self.model_key = "vascular"
        self.mesh_cache = mesh_cache       # 可选 — 提供则共享 polydata

    def load(self, model_path) -> bool:
        """加载血管 STL 模型；若 mesh_cache 提供则共享 PolyData，避免重复 pv.read。"""
        path = Path(model_path)
        if not path.exists():
            self.view.status_changed.emit(f"[VesselRenderer] Model not found: {path}")
            return False
        cached_mesh = None
        if self.mesh_cache is not None:
            try:
                cached_mesh = self.mesh_cache.get_normalized(
                    path, normalized_length=self.view.DEFAULT_NORMALIZED_LENGTH,
                )
            except Exception:
                cached_mesh = None
        # 真实血管 / 内窥镜风格颜色 — 暗红肉色
        self.actor = self.view.load_model(
            path,
            model_key=self.model_key,
            position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            scale=self.view.DEFAULT_MODEL_SCALE,
            color="#bf4038",        # 接近 (0.75, 0.25, 0.22)
            opacity=0.45,
            reset_camera=True,
            mesh=cached_mesh,
            specular=0.2,
            ambient=0.25,
            diffuse=0.7,
        )
        if self.actor is not None:
            # 关闭背面剔除 — 第一人称镜头进入 lumen 时可看到内壁
            try:
                self.actor.GetProperty().SetBackfaceCulling(0)
                self.actor.GetProperty().SetFrontfaceCulling(0)
            except Exception:
                pass
        return self.actor is not None

    def show(self) -> None:
        """显示血管模型。"""
        if self.actor is not None:
            self.actor.SetVisibility(True)
            self.view.plotter.render()

    def hide(self) -> None:
        """隐藏血管模型。"""
        if self.actor is not None:
            self.actor.SetVisibility(False)
            self.view.plotter.render()

    def set_opacity(self, opacity: float) -> None:
        """调整血管透明度。"""
        self.view.set_model_opacity(self.model_key, opacity)

    def toggle_wireframe(self) -> bool:
        """切换实体表面 ↔ 线框（VTK Representation）；返回切后是否为线框。"""
        if self.actor is None:
            return False
        try:
            prop = self.actor.GetProperty()
            # vtkProperty Representation: 0=Points, 1=Wireframe, 2=Surface
            is_wireframe_now = (prop.GetRepresentation() == 1)
            if is_wireframe_now:
                prop.SetRepresentationToSurface()
                new_state = False
            else:
                prop.SetRepresentationToWireframe()
                new_state = True
            self.view.plotter.render()
            return new_state
        except Exception:
            return False

    def clear(self) -> None:
        """从视窗中移除血管模型。"""
        if self.actor is not None:
            self.view.plotter.remove_actor(self.actor)
            self.actor = None
            self.view.plotter.render()
