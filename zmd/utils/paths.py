"""统一管理项目路径，避免各处写相对路径。"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT_DIR / "assets"
CONFIG_DIR = ROOT_DIR / "config"
PROJECTS_DIR = ROOT_DIR / "projects"
SCRIPTS_DIR = ROOT_DIR / "scripts"
TESTS_DIR = ROOT_DIR / "tests"

MODELS_DIR = ASSETS_DIR / "models"
VESSELS_DIR = MODELS_DIR / "vessels"
ROBOTS_DIR = MODELS_DIR / "robots"
