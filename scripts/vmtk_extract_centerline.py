"""VMTK 中心线提取脚本 — 多入口 / 多出口非交互管线。

输入：assets/models/vessels/VascuModel.stl
输出：
    projects/cache/VascuModel_capped.vtp     封口后的预处理曲面（调试用）
    projects/cache/VascuModel_centerline.vtp 中心线树（2 sources × 3 targets）

实现要点：
    1. 读取 STL，仅保留最大连通域，清理 + 三角化（不降采样以保几何精度）。
    2. 用 vmtkSurfaceCapper(Interactive=0, Method='centerpoint') 自动封口；
       这一步替代 vmtkCenterlines 内部 capping 的交互提问，避免阻塞。
    3. 用 vmtkCenterlines + pointlist seed selector 传入项目配置中的
       2 个入口 + 3 个出口，得到完整的分支树并附 MaximumInscribedSphereRadius。

运行：
    set ZMD_VMTK_ENV=D:\softer\miniconda3\envs\vmtk-env
    python scripts/vmtk_extract_centerline.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

HEAD = "=" * 60
DEFAULT_VMTK_ENV = Path(
    os.environ.get("ZMD_VMTK_ENV", r"D:\softer\miniconda3\envs\vmtk-env")
)


# ----------------------------------------------------------------------
# 入口：找到 VMTK Python，转发执行
# ----------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_stl = root / "assets" / "models" / "vessels" / "VascuModel.stl"
    cache_dir = root / "projects" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    capped_vtp = cache_dir / "VascuModel_capped.vtp"
    centerline_vtp = cache_dir / "VascuModel_centerline.vtp"

    if not input_stl.exists():
        print(f"[ERROR] STL not found: {input_stl}")
        sys.exit(1)

    print(HEAD)
    print("  step 1/4  Checking VMTK availability ...")

    try:
        import vmtk  # noqa: F401
    except ImportError:
        env_python = find_configured_vmtk_python()
        if env_python and env_python.resolve() != Path(sys.executable).resolve():
            print(f"  current Python has no VMTK, relaunching with: {env_python}")
            result = subprocess.run([str(env_python), str(Path(__file__).resolve())])
            sys.exit(result.returncode)
        conda_exe = shutil.which("conda") or str(
            DEFAULT_VMTK_ENV.parent.parent / "Scripts" / "conda.exe"
        )
        if conda_exe and Path(conda_exe).exists():
            result = subprocess.run([
                conda_exe, "run", "-n", DEFAULT_VMTK_ENV.name,
                "python", str(Path(__file__).resolve()),
            ])
            sys.exit(result.returncode)
        print(HEAD)
        print("  !!  VMTK not available.")
        print(f"  Expected env: {DEFAULT_VMTK_ENV}")
        print("  Or set ZMD_VMTK_ENV / ZMD_VMTK_PYTHON before running.")
        sys.exit(1)

    print("  vmtk Python module found, running pipeline ...")
    run_pipeline(input_stl, capped_vtp, centerline_vtp, root)


def find_configured_vmtk_python() -> Optional[Path]:
    candidates: List[Path] = []
    if os.environ.get("ZMD_VMTK_PYTHON"):
        candidates.append(Path(os.environ["ZMD_VMTK_PYTHON"]))
    candidates.extend([
        DEFAULT_VMTK_ENV / "python.exe",
        DEFAULT_VMTK_ENV / "Scripts" / "python.exe",
        DEFAULT_VMTK_ENV / "bin" / "python",
    ])
    for python_path in candidates:
        if python_path.exists():
            return python_path
    return None


# ----------------------------------------------------------------------
# 核心管线
# ----------------------------------------------------------------------

def load_project_endpoints(root: Path) -> Tuple[List[Tuple[float, float, float]],
                                                List[Tuple[float, float, float]]]:
    """读取 scene_config 中所有入口 / 出口坐标。"""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from zmd.scene.scene_config import (
        DEFAULT_ENTRY_POINT_PRESETS,
        DEFAULT_EXIT_POINT_PRESETS,
    )
    inlets = [tuple(float(v) for v in p["entry_point"]) for p in DEFAULT_ENTRY_POINT_PRESETS]
    outlets = [tuple(float(v) for v in p["exit_point"]) for p in DEFAULT_EXIT_POINT_PRESETS]
    return inlets, outlets


def normalize_to_display_frame(surface):
    """把 STL 变换到项目最终显示坐标系，便于复用 scene_config 中的入口/出口。

    与 zmd.rendering.pyvista_view.PyVistaView.prepare_mesh + actor.SetScale(0.1)
    的等价变换：translate(-center) → scale(900/length) → scale(0.1)。
    """
    import vtk

    b = surface.GetBounds()
    center = ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
    diag = ((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2) ** 0.5
    scale = (900.0 / diag) * 0.1 if diag > 0 else 1.0

    tform = vtk.vtkTransform()
    tform.PostMultiply()
    tform.Translate(-center[0], -center[1], -center[2])
    tform.Scale(scale, scale, scale)

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(surface)
    tf.SetTransform(tform)
    tf.Update()
    out = tf.GetOutput()
    nb = out.GetBounds()
    print(f"  [normalized] center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
          f"scale={scale:.4f}")
    print(f"  [normalized] bounds=({nb[0]:.2f},{nb[1]:.2f},{nb[2]:.2f},"
          f"{nb[3]:.2f},{nb[4]:.2f},{nb[5]:.2f})")
    return out


def extract_inner_wall(surface, max_lumen_diameter: float = 20.0):
    """从双层壁封闭流形中只保留内壁三角面。

    思路：原 STL 是 inner + outer + end-cap 焊合的单连通闭合流形，vmtkCenterlines
    在其"内部"（壁体内）求 Voronoi 中心线，因此跑在内外壁之间而不是 lumen 里。

    每个三角面用 vtkOBBTree 沿其向外法向投射射线：
      - 内壁 cell：射线穿过 lumen 命中对侧内壁，距离 ≈ lumen 直径（finite，<max_lumen_diameter）
      - 外壁 cell：射线飞向空间，无命中 / 距离 >> max_lumen_diameter
      - 端面 cap：射线沿血管轴方向，命中另一端 cap，距离 ≈ vessel length（>max_lumen_diameter）
    保留内壁 cell 后端口会重新出现孔洞，由后续 vmtkSurfaceCapper 重封。
    """
    import vtk
    import numpy as np

    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(surface)
    nf.ComputeCellNormalsOn()
    nf.ComputePointNormalsOff()
    nf.ConsistencyOn()
    nf.AutoOrientNormalsOn()
    nf.NonManifoldTraversalOn()
    nf.Update()
    surf = nf.GetOutput()

    normals = surf.GetCellData().GetNormals()
    n_cells = surf.GetNumberOfCells()

    obb = vtk.vtkOBBTree()
    obb.SetDataSet(surf)
    obb.BuildLocator()

    keep = vtk.vtkIdTypeArray()
    keep.SetNumberOfComponents(1)

    inter_pts = vtk.vtkPoints()
    progress_step = max(1, n_cells // 20)
    inner = 0
    for cid in range(n_cells):
        if cid % progress_step == 0:
            print(f"    classify {cid}/{n_cells} kept={inner}", flush=True)
        cell = surf.GetCell(cid)
        pts = cell.GetPoints()
        cnt = np.zeros(3)
        for i in range(3):
            cnt += np.array(pts.GetPoint(i))
        cnt /= 3.0
        nrm = np.array(normals.GetTuple3(cid))
        start = cnt + 1e-4 * nrm
        end = cnt + (max_lumen_diameter + 1.0) * nrm
        inter_pts.Reset()
        obb.IntersectWithLine(start.tolist(), end.tolist(), inter_pts, None)
        if inter_pts.GetNumberOfPoints() > 0:
            d = np.linalg.norm(np.array(inter_pts.GetPoint(0)) - cnt)
            if d <= max_lumen_diameter:
                keep.InsertNextValue(cid)
                inner += 1

    print(f"  [inner-wall] kept {inner}/{n_cells} cells "
          f"(ratio {inner/n_cells*100:.1f}%)")

    sel = vtk.vtkSelectionNode()
    sel.SetFieldType(vtk.vtkSelectionNode.CELL)
    sel.SetContentType(vtk.vtkSelectionNode.INDICES)
    sel.SetSelectionList(keep)
    selection = vtk.vtkSelection()
    selection.AddNode(sel)
    extract = vtk.vtkExtractSelection()
    extract.SetInputData(0, surf)
    extract.SetInputData(1, selection)
    extract.Update()

    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(extract.GetOutput())
    geom.Update()
    inner_surface = geom.GetOutput()

    # 内壁可能多碎片 — 取最大连通域
    cf = vtk.vtkPolyDataConnectivityFilter()
    cf.SetInputData(inner_surface)
    cf.SetExtractionModeToLargestRegion()
    cf.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(cf.GetOutput())
    clean.Update()
    return clean.GetOutput()


def preprocess_surface(surface):
    """归一化 → 清理 → 三角化 → 取最大连通域 → 仅保留内壁。"""
    import vtk
    from vmtk import vmtkscripts

    surface = normalize_to_display_frame(surface)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.Update()
    surface = cleaner.GetOutput()

    tri = vmtkscripts.vmtkSurfaceTriangle()
    tri.Surface = surface
    tri.Execute()
    surface = tri.Surface

    conn = vmtkscripts.vmtkSurfaceConnectivity()
    conn.Surface = surface
    conn.CleanOutput = 1
    conn.Execute()
    surface = conn.Surface
    print(f"  [largest cc] {surface.GetNumberOfPoints()} pts, {surface.GetNumberOfCells()} cells")

    surface = extract_inner_wall(surface)
    print(f"  [inner-only] {surface.GetNumberOfPoints()} pts, {surface.GetNumberOfCells()} cells")
    return surface


def cap_surface(surface):
    """非交互封口 — 用 centerpoint 方法关闭所有孔洞。"""
    from vmtk import vmtkscripts

    capper = vmtkscripts.vmtkSurfaceCapper()
    capper.Surface = surface
    capper.Method = "centerpoint"   # 自动用孔洞中心点封口，无需交互
    capper.Interactive = 0
    capper.TriangleOutput = 1
    capper.Execute()
    print(f"  [capped] {capper.Surface.GetNumberOfPoints()} pts, "
          f"{capper.Surface.GetNumberOfCells()} cells")
    return capper.Surface


def extract_centerline_tree(surface, sources, targets, output_vtp: Path) -> bool:
    """对每个 source 跑一次 vmtkCenterlines（到全部 targets），最后合并所有分支。

    VMTK 在多 source + 多 target 配置下，会把每个 target 连到“最近的” source，
    因此远端入口可能完全得不到中心线。改成 per-source 多次提取后 append，可保证
    所有 (source, target) 组合都得到一条路径。
    """
    import vtk
    from vmtk import vmtkscripts

    flat = lambda pts: [v for p in pts for v in p]
    append = vtk.vtkAppendPolyData()
    total_lines = 0

    for s_idx, src in enumerate(sources):
        cl = vmtkscripts.vmtkCenterlines()
        cl.Surface = surface
        cl.SeedSelectorName = "pointlist"
        cl.SourcePoints = list(src)
        cl.TargetPoints = flat(targets)
        cl.AppendEndPoints = 1
        cl.RadiusArrayName = "MaximumInscribedSphereRadius"
        cl.Resampling = 1
        cl.ResamplingStepLength = 0.5
        try:
            cl.Execute()
        except Exception as e:
            print(f"  vmtkCenterlines FAILED for source {s_idx}: {e}")
            return False
        out = cl.Centerlines
        if out is None or out.GetNumberOfPoints() < 2:
            print(f"  [ERROR] no centerline from source {s_idx} ({src}).")
            return False
        print(f"  source #{s_idx} {src}: {out.GetNumberOfPoints()} pts, "
              f"{out.GetNumberOfLines()} lines")
        total_lines += out.GetNumberOfLines()
        append.AddInputData(out)

    append.Update()
    merged = append.GetOutput()
    print(f"  merged: {merged.GetNumberOfPoints()} pts, {merged.GetNumberOfLines()} lines")

    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.Surface = merged
    writer.OutputFileName = str(output_vtp)
    writer.Execute()
    return True


def write_surface(surface, path: Path) -> None:
    from vmtk import vmtkscripts

    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.Surface = surface
    writer.OutputFileName = str(path)
    writer.Execute()


def verify_output(vtp_path: Path) -> None:
    print(f"\n{HEAD}")
    print("  step 4/4  Verifying output ...\n")

    import vtk

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    mesh = reader.GetOutput()
    bounds = mesh.GetBounds()
    point_data = mesh.GetPointData()
    arrays = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]

    print(f"  Points    : {mesh.GetNumberOfPoints()}")
    print(f"  Lines     : {mesh.GetNumberOfLines()}")
    print(f"  Bounds    : {tuple(round(v, 1) for v in bounds)}")
    print(f"  Arrays    : {arrays}")

    radius = point_data.GetArray("MaximumInscribedSphereRadius")
    if radius is not None:
        values = [radius.GetValue(i) for i in range(radius.GetNumberOfTuples())]
        print(
            "  Radius    : "
            f"min={min(values):.2f}, max={max(values):.2f}, mean={sum(values) / len(values):.2f}"
        )
    print(HEAD)


def run_pipeline(input_stl: Path, capped_vtp: Path, centerline_vtp: Path, root: Path) -> None:
    from vmtk import vmtkscripts

    print(f"\n{HEAD}")
    print("  step 2/4  Loading and preprocessing surface ...\n")

    if capped_vtp.exists() and os.environ.get("ZMD_VMTK_FORCE") != "1":
        print(f"  Reusing cached capped surface: {capped_vtp} "
              f"(set ZMD_VMTK_FORCE=1 to rebuild)")
        reader = vmtkscripts.vmtkSurfaceReader()
        reader.InputFileName = str(capped_vtp)
        reader.Execute()
        surface = reader.Surface
    else:
        reader = vmtkscripts.vmtkSurfaceReader()
        reader.InputFileName = str(input_stl)
        reader.Execute()
        surface = reader.Surface
        print(f"  raw STL: {surface.GetNumberOfPoints()} pts, "
              f"{surface.GetNumberOfCells()} cells")
        surface = preprocess_surface(surface)
        surface = cap_surface(surface)
        write_surface(surface, capped_vtp)
        print(f"  Capped surface saved to: {capped_vtp}")

    inlets, outlets = load_project_endpoints(root)
    print(f"\n  Inlets  ({len(inlets)}) : {inlets}")
    print(f"  Outlets ({len(outlets)}): {outlets}")

    print(f"\n{HEAD}")
    print("  step 3/4  Extracting centerline tree ...\n")
    ok = extract_centerline_tree(surface, inlets, outlets, centerline_vtp)
    if not ok:
        sys.exit(1)

    verify_output(centerline_vtp)


if __name__ == "__main__":
    main()
