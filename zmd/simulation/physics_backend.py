"""物理后端工厂，根据配置创建合适的后端实例。"""
from typing import Dict

from zmd.simulation.backends.base_backend import BasePhysicsBackend
from zmd.simulation.backends.dummy_backend import DummyPhysicsBackend


def create_physics_backend(config: Dict) -> BasePhysicsBackend:
    """根据配置字典创建物理后端。"""
    backend_name = config.get("backend", "dummy")
    if backend_name == "dummy":
        return DummyPhysicsBackend()
    # TODO: 后续接入 Bullet/MuJoCo 等真实物理引擎
    raise ValueError(f"Unsupported physics backend: {backend_name}")
