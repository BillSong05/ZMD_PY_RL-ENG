"""场景配置 — 管理 4 场景 2×2 布局和模型变换默认值。"""
from typing import Dict, List, Tuple

DEFAULT_MODEL_TRANSFORMS: Dict[str, Dict] = {
    "vascular": {"position": (0.0, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0)},
    "robot": {"position": (-11.2, -78.0, -16.65), "rotation": (-90.0, 0.0, -90.0)},
}

DEFAULT_ENTRY_POINT_PRESETS: List[Dict] = [
    {"label": "Entry 1", "entry_point": (-11.2, -39.0, 3.0), "robot_position": (0.0, 0.0, 0.0)},
    {"label": "Entry 2", "entry_point": (5.8, -39.0, 3.0), "robot_position": (17.0, 0.0, 0.0)},
]

DEFAULT_EXIT_POINT_PRESETS: List[Dict] = [
    # pullback_fraction:沿所选分支中心线往回退的比例(默认 0.10)。
    # 出口 3 比另外两条短分支多退 3%(13%),弯口处不要顶到分叉边界。
    {"label": "Exit 1", "exit_point": (-18.0, 31.7, 2.8), "pullback_fraction": 0.19},
    {"label": "Exit 2", "exit_point": (18.0, 30.9, 0.2),  "pullback_fraction": 0.17},
    {"label": "Exit 3", "exit_point": (4.1, 38.9, 2.8),   "pullback_fraction": 0.12},
]

DEFAULT_VASCULAR_SETTINGS: Dict = {
    "opacity": 0.45,
    "entry_point": DEFAULT_ENTRY_POINT_PRESETS[0]["entry_point"],
    "exit_point": DEFAULT_EXIT_POINT_PRESETS[2]["exit_point"],
    "marker_radius": 0.3,
}

DEFAULT_CENTERLINE_SETTINGS: Dict = {
    "radius": 0.10,
    "visible": True,
    "show_all_branches": True,
}

DEFAULT_GUIDEWIRE_SETTINGS: Dict = {
    "radius": 0.01,        # 中心线导丝默认 — 极细 baseline
    "step": 3,
}

# 按类型的默认半径
GUIDEWIRE_DEFAULT_RADIUS: Dict[str, float] = {
    "mujoco": 0.1,
}

SCENE_LAYOUTS: Dict[str, List[Tuple[str, int, int, int]]] = {
    "four_view": [
    ("Experiment Scene 1", 0, 0, 0),
    ("Experiment Scene 2", 1, 0, 1),
    ("Experiment Scene 3", 2, 1, 0),
    ("Experiment Scene 4", 3, 1, 1),
    ],
}
