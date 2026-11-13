"""STL 模型加载工具。"""
from pathlib import Path

import pyvista as pv


def load_stl_mesh(stl_path: Path) -> pv.PolyData:
    """加载 STL 文件并返回 PyVista PolyData。"""
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    return pv.read(stl_path)


def normalize_mesh(mesh: pv.PolyData, target_length: float = 900.0) -> pv.PolyData:
    """将 mesh 居中并缩放到目标长度。"""
    mesh = mesh.copy()
    mesh.translate((-mesh.center[0], -mesh.center[1], -mesh.center[2]), inplace=True)
    if mesh.length > 0:
        mesh.scale(target_length / mesh.length, inplace=True)
    return mesh
