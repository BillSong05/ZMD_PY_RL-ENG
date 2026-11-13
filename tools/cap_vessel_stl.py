"""把血管 STL 的所有开口（inlet/outlet）封口成闭合 manifold。

- 输入: projects/cache/VascuModel_collision.stl （525 条 boundary edges）
- 处理: vtkFillHolesFilter 通过给定 hole_size 把所有边界环闭合
- 保险: 用 pymeshfix 做最终修复（去除 self-intersection / 非流形 / 残余孔洞）
- 输出: 覆盖原文件，并备份原文件为 *_open_backup.stl
- 校验: 输出 boundary_edges==0 时通过

运行::

    python tools/cap_vessel_stl.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "projects" / "cache" / "VascuModel_collision.stl"
BACKUP = ROOT / "projects" / "cache" / "VascuModel_collision_open_backup.stl"

# 血管 bounds ~ 36 × 79 × 22；inlet/outlet 直径估计 < 8。
# 给 30 让 fill_holes 把所有 inlet/outlet 都吃掉。
HOLE_SIZE = 30.0


def boundary_edge_count(mesh: pv.PolyData) -> int:
    e = mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False)
    return int(e.n_cells)


def fill_with_vtk(mesh: pv.PolyData, hole_size: float) -> pv.PolyData:
    out = mesh.fill_holes(hole_size)
    if isinstance(out, pv.PolyData):
        return out.triangulate().clean()
    return pv.PolyData(out).triangulate().clean()


def fill_with_pymeshfix(mesh: pv.PolyData) -> pv.PolyData:
    try:
        import pymeshfix
    except ImportError:
        print("[INFO] pymeshfix 未安装，跳过保险修复 — pip install pymeshfix 可启用")
        return mesh
    mf = pymeshfix.MeshFix(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    mf.repair(joincomp=False, remove_smallest_components=False)
    pts = np.asarray(mf.points)
    faces = np.asarray(mf.faces)
    faces_pv = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64),
                          faces.astype(np.int64)])
    return pv.PolyData(pts, faces_pv).clean()


def main() -> int:
    if not TARGET.exists():
        print(f"[FATAL] 找不到 {TARGET}")
        return 1

    # 如果备份已存在（说明前一次跑过、原文件被覆盖了），从备份恢复后再处理
    if BACKUP.exists():
        src = BACKUP
        print(f"从备份恢复源文件: {BACKUP}")
    else:
        src = TARGET
        shutil.copy2(TARGET, BACKUP)
        print(f"备份原文件 -> {BACKUP}")

    mesh = pv.read(str(src))
    n0 = boundary_edge_count(mesh)
    print(f"源 STL: tris={mesh.n_cells}  pts={mesh.n_points}  boundary_edges={n0}")
    if n0 == 0:
        print("已经闭合，无需处理。")
        return 0

    # 先做一次 clean — 合并重合顶点，避免 T-junction / 重复点导致 boundary 错乱
    mesh = mesh.clean(tolerance=1e-6).triangulate()
    print(f"clean 后: tris={mesh.n_cells}  pts={mesh.n_points}  boundary_edges={boundary_edge_count(mesh)}")

    # 步骤 1: pymeshfix 主修复（对非流形 / T-junction / 多边界环最稳）
    filled = fill_with_pymeshfix(mesh)
    n1 = boundary_edge_count(filled)
    print(f"pymeshfix: tris={filled.n_cells}  pts={filled.n_points}  boundary_edges={n1}")

    # 步骤 2: 若还有残余孔洞，再走 VTK fill_holes 用大 hole_size 兜底
    if n1 > 0:
        for hs in (HOLE_SIZE, HOLE_SIZE * 5, HOLE_SIZE * 30):
            filled2 = fill_with_vtk(filled, hs)
            n2 = boundary_edge_count(filled2)
            print(f"vtkFillHoles(size={hs}): boundary_edges={n2}")
            filled = filled2
            if n2 == 0:
                break

    final_boundary = boundary_edge_count(filled)
    if final_boundary != 0:
        print(f"[FAIL] 仍有 {final_boundary} 条 boundary edge，未完全闭合")
        return 1

    # 法向一致化
    filled = filled.compute_normals(
        inplace=False, auto_orient_normals=True, consistent_normals=True,
        flip_normals=False, non_manifold_traversal=False)
    filled.save(str(TARGET))
    re = pv.read(str(TARGET))
    print(f"\n写回 {TARGET}")
    print(f"  tris={re.n_cells}  pts={re.n_points}  boundary_edges={boundary_edge_count(re)}")
    print(f"  bounds={list(re.bounds)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
