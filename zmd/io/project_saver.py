"""工程保存（占位，后续实现）。"""
from pathlib import Path
from zmd.io.project_loader import ProjectConfig


def save_project_config(config: ProjectConfig, output_path: Path):
    """将 ProjectConfig 保存为 YAML 文件。"""
    # TODO: 接入 PyYAML 实现
    raise NotImplementedError("Project saving not yet implemented")
