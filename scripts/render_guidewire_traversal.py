"""离屏渲染：导丝沿 (IN1→OUT3) 默认路径推进的多帧截图。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zmd.centerline.centerline_model import Centerline  # noqa: E402
from zmd.scene.scene_config import (  # noqa: E402
    DEFAULT_ENTRY_POINT_PRESETS,
    DEFAULT_EXIT_POINT_PRESETS,
    DEFAULT_VASCULAR_SETTINGS,
)

CACHE = ROOT / "projects" / "cache"
VESSEL_STL = ROOT / "assets" / "models" / "vessels" / "VascuModel.stl"
VTP = CACHE / "VascuModel_centerline.vtp"

FRACTIONS = [0.0, 0.33, 0.66, 1.0]


def normalize_vessel(stl_path: Path) -> pv.PolyData:
    m = pv.read(stl_path).copy()
    m.translate((-m.center[0], -m.center[1], -m.center[2]), inplace=True)
    if m.length > 0:
        m.scale(900.0 / m.length, inplace=True)
    m.scale(0.1, inplace=True)
    return m


def main() -> int:
    cl = Centerline.load_from_vtp(
        VTP,
        source_point=DEFAULT_VASCULAR_SETTINGS["entry_point"],
        target_point=DEFAULT_VASCULAR_SETTINGS["exit_point"],
    )
    if cl is None:
        print("[ERROR] failed to load centerline")
        return 1
    print(f"  selected route: {cl.n_points} pts (IN1 -> OUT3 default)")

    vessel = normalize_vessel(VESSEL_STL)
    inlets = [np.asarray(p["entry_point"]) for p in DEFAULT_ENTRY_POINT_PRESETS]
    outlets = [np.asarray(p["exit_point"]) for p in DEFAULT_EXIT_POINT_PRESETS]

    pts = np.asarray(cl.points)
    for frac in FRACTIONS:
        idx = max(2, int(round(frac * (len(pts) - 1))) + 1)
        visible = pts[:idx]
        tip = visible[-1]

        plotter = pv.Plotter(off_screen=True, window_size=(1280, 960))
        plotter.set_background("white")
        plotter.add_mesh(vessel, color="#dddddd", opacity=0.16, smooth_shading=True)

        # 灰色淡色显示所有分支
        for b in cl.branches:
            bpts = np.asarray(b.points)
            bp = pv.PolyData(bpts)
            bp.lines = np.hstack([[len(bpts)], np.arange(len(bpts))])
            plotter.add_mesh(bp.tube(radius=0.08, n_sides=10),
                             color="#94a3b8", opacity=0.4)

        # 当前选中路径 — 琥珀色
        ap = pv.PolyData(pts)
        ap.lines = np.hstack([[len(pts)], np.arange(len(pts))])
        plotter.add_mesh(ap.tube(radius=0.13, n_sides=12), color="#f59e0b", opacity=0.85)

        # 导丝 — 蓝色管
        gw = pv.PolyData(visible)
        gw.lines = np.hstack([[len(visible)], np.arange(len(visible))])
        plotter.add_mesh(gw.tube(radius=0.20, n_sides=14), color="#2563eb")
        plotter.add_mesh(pv.Sphere(radius=0.45, center=tip), color="#1d4ed8")

        for i, p in enumerate(inlets, 1):
            plotter.add_mesh(pv.Sphere(radius=1.2, center=p), color="#16a34a")
            plotter.add_point_labels([p.tolist()], [f"IN{i}"], font_size=16,
                                     text_color="#065f46", shape=None,
                                     always_visible=True, show_points=False)
        for i, p in enumerate(outlets, 1):
            plotter.add_mesh(pv.Sphere(radius=1.2, center=p), color="#f97316")
            plotter.add_point_labels([p.tolist()], [f"OUT{i}"], font_size=16,
                                     text_color="#9a3412", shape=None,
                                     always_visible=True, show_points=False)

        plotter.add_axes()
        plotter.view_vector((1, -1.4, 0.7))
        plotter.reset_camera()
        out_png = CACHE / f"guidewire_traversal_{int(frac * 100):03d}.png"
        plotter.screenshot(str(out_png))
        plotter.close()
        print(f"  wrote {out_png}  (frac={frac:.2f}, tip=({tip[0]:.1f},{tip[1]:.1f},{tip[2]:.1f}))")

    return 0


if __name__ == "__main__":
    sys.exit(main())
