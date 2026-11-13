"""仿真引擎，封装物理后端，提供统一 step/reset 接口。"""
from typing import Any, Dict, Optional

from zmd.simulation.backends.base_backend import BasePhysicsBackend


class SimulationEngine:
    """仿真引擎，UI/RL 通过此接口与物理后端交互。"""

    def __init__(self, backend: BasePhysicsBackend):
        self.backend = backend

    def reset(self, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.backend.reset(initial_state)

    def step(self, action: Dict[str, Any], dt: float = 0.01) -> Dict[str, Any]:
        return self.backend.step(action, dt)

    def get_state(self) -> Dict[str, Any]:
        return self.backend.get_state()

    def close(self) -> None:
        self.backend.close()
