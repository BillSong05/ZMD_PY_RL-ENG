"""工程配置定义与加载。

全部工程配置由 YAML 文件驱动，默认工程和自定义工程统一使用 ProjectConfig。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from zmd.utils.paths import PROJECTS_DIR, ROOT_DIR


@dataclass
class ProjectConfig:
    """统一工程配置，默认和自定义工程均由此 dataclass 描述。"""
    name: str
    vessel_model_path: Path
    robot_model_path: Optional[Path] = None
    guidewire_config: Optional[Dict[str, Any]] = None
    physics_config: Optional[Dict[str, Any]] = None
    scene_layout: str = "four_view"
    compact_vascular_controls: bool = True


def _resolve_model_path(raw_path: str) -> Path:
    """将 YAML 中的相对路径基于项目根目录解析为绝对路径。"""
    path = Path(raw_path)
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


def load_project_config(yaml_path: str | Path) -> ProjectConfig:
    """从 YAML 文件加载工程配置。

    Args:
        yaml_path: YAML 配置文件路径（绝对或相对于项目根）。

    Returns:
        解析后的 ProjectConfig。

    Raises:
        FileNotFoundError: YAML 文件或 vessel_model 不存在。
    """
    path = Path(yaml_path)
    if not path.is_absolute():
        path = ROOT_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Project configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    vessel_path = _resolve_model_path(data["vessel_model"])
    if not vessel_path.exists():
        raise FileNotFoundError(
            f"Vessel model file for project "
            f"[{data.get('name', 'unnamed')}] not found: {vessel_path}"
        )

    robot_path: Optional[Path] = None
    if data.get("robot_model"):
        robot_path = _resolve_model_path(data["robot_model"])
        if not robot_path.exists():
            raise FileNotFoundError(
            f"Robot model file for project "
            f"[{data.get('name', 'unnamed')}] not found: {robot_path}"
            )

    return ProjectConfig(
        name=data.get("name", "unnamed"),
        vessel_model_path=vessel_path,
        robot_model_path=robot_path,
        scene_layout=data.get("scene_layout", "four_view"),
        compact_vascular_controls=data.get("compact_vascular_controls", True),
        physics_config=data.get("physics"),
        guidewire_config=data.get("guidewire"),
    )


def make_default_project_config() -> ProjectConfig:
    """加载默认工程配置。

    优先读取 projects/default_project.yaml；
    如果文件不存在则回退到内置硬编码默认值。
    """
    default_yaml = PROJECTS_DIR / "default_project.yaml"
    if default_yaml.exists():
        return load_project_config(default_yaml)
    # 极端 fallback —— 配置文件丢失时的兜底
    from zmd.utils.paths import ROBOTS_DIR, VESSELS_DIR

    return ProjectConfig(
        name="default_vascular_project",
        vessel_model_path=VESSELS_DIR / "VascuModel.stl",
        robot_model_path=ROBOTS_DIR / "VascubotModel.stl",
        scene_layout="four_view",
        compact_vascular_controls=True,
    )
