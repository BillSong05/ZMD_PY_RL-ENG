"""离屏渲染验证截图。

把 VMTK 输出的中心线树、血管 STL（半透明）和 2 入口 / 3 出口标记
一起渲染到 projects/cache/centerline_verify_*.png，多视角各一张。
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zmd.scene.scene_config import (  # noqa: E402
    DEFAULT_ENTRY_POINT_PRESETS,
    DEFAULT_EXIT_POINT_PRESETS,
)

CACHE = ROOT / "projects" / "cache"
VESSEL_STL = ROOT / "assets" / "models" / "vessels" / "VascuModel.stl"
CENTERLINE_VTP = CACHE / "VascuModel_centerline.vtp"

VIEWS = {
    "iso": (1, -1.4, 0.7),
    "front": (0, -1, 0),
    "top": (0, 0.05, 1),
    "side": (1, 0, 0),
}


def main() -> int:
    if not CENTERLINE_VTP.exists():
        print(f"[ERROR] centerline VTP not found: {CENTERLINE_VTP}")
        return 1

    vessel = pv.read(VESSEL_STL)
    # 与 zmd.rendering.pyvista_view 同样的归一化：translate(-center) → scale(900/length) → 0.1
    vessel = vessel.copy()
    vessel.translate((-vessel.center[0], -vessel.center[1], -vessel.center[2]), inplace=True)
    if vessel.length > 0:
        vessel.scale(900.0 / vessel.length, inplace=True)
    vessel.scale(0.1, inplace=True)

    centerline = pv.read(CENTERLINE_VTP)

    print(f"  vessel    : {vessel.n_points} pts, {vessel.n_cells} cells")
    print(f"  centerline: {centerline.n_points} pts, {centerline.n_cells} lines")
    radius_arr = None
    for name in centerline.point_data.keys():
        if "Radius" in name or "radius" in name:
            radius_arr = name
            break
    print(f"  radius    : {radius_arr}")

    inlets = [np.asarray(p["entry_point"]) for p in DEFAULT_ENTRY_POINT_PRESETS]
    outlets = [np.asarray(p["exit_point"]) for p in DEFAULT_EXIT_POINT_PRESETS]

    written: list[Path] = []
    for name, direction in VIEWS.items():
        out_png = CACHE / f"centerline_verify_{name}.png"
        plotter = pv.Plotter(off_screen=True, window_size=(1280, 960))
        plotter.set_background("white")
        plotter.add_mesh(vessel, color="#dddddd", opacity=0.18, smooth_shading=True)

        # centerline tube — 用半径数组生成沿线管，能直观看出血管粗细
        if radius_arr is not None:
            centerline.set_active_scalars(radius_arr)
            tubed = centerline.tube(scalars=radius_arr, radius_factor=1.0, n_sides=12)
            plotter.add_mesh(tubed, color="#e11d48", smooth_shading=True)
        else:
            tubed = centerline.tube(radius=0.3, n_sides=12)
            plotter.add_mesh(tubed, color="#e11d48", smooth_shading=True)

        for i, p in enumerate(inlets, 1):
            plotter.add_mesh(pv.Sphere(radius=1.5, center=p), color="#16a34a")
            plotter.add_point_labels(
                [p.tolist()], [f"IN{i}"], font_size=18, text_color="#065f46",
                shape=None, always_visible=True, show_points=False,
            )
        for i, p in enumerate(outlets, 1):
            plotter.add_mesh(pv.Sphere(radius=1.5, center=p), color="#f97316")
            plotter.add_point_labels(
                [p.tolist()], [f"OUT{i}"], font_size=18, text_color="#9a3412",
                shape=None, always_visible=True, show_points=False,
            )

        plotter.add_axes()
        plotter.view_vector(direction)
        plotter.reset_camera()
        plotter.screenshot(str(out_png))
        plotter.close()
        written.append(out_png)
        print(f"  wrote {out_png}")

    print("\nDone:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
