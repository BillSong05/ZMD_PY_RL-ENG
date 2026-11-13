"""物理后端抽象基类。"""
from typing import Any, Dict, Optional


class BasePhysicsBackend:
    """物理仿真后端接口，后续可接入 Bullet/PhysX/MuJoCo 等。"""

    def reset(self, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """重置仿真到初始状态。"""
        raise NotImplementedError

    def step(self, action: Dict[str, Any], dt: float = 0.01) -> Dict[str, Any]:
        """执行一步仿真。"""
        raise NotImplementedError

    def get_state(self) -> Dict[str, Any]:
        """获取当前仿真状态。"""
        raise NotImplementedError

    def close(self) -> None:
        """释放后端资源。"""
        pass
