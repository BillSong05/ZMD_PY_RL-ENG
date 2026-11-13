"""统一模型加载入口，根据类型分发到对应加载器。"""
from pathlib import Path
from typing import Any, Dict

import pyvista as pv

from zmd.io.stl_loader import load_stl_mesh, normalize_mesh


def load_model(model_path: Path, model_type: str = "stl") -> pv.PolyData:
    """加载模型文件并返回标准化后的 mesh。"""
    if model_type == "stl":
        mesh = load_stl_mesh(model_path)
        return normalize_mesh(mesh)
    raise ValueError(f"Unsupported model type: {model_type}")


def get_model_info(mesh: pv.PolyData) -> Dict[str, Any]:
    """提取模型基本信息。"""
    return {
        "n_points": mesh.n_points,
        "n_cells": mesh.n_cells,
        "bounds": mesh.bounds,
        "center": mesh.center,
        "length": mesh.length,
    }
