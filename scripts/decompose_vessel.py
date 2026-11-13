"""把血管 collision STL 拆成若干凸包 STL,供 MuJoCo `<mesh>` 资产逐个加载。

凸分解后,凹陷区域也能产生正确碰撞 — cathsim 用的就是这条路。
没装 coacd / pyvhacd 时,自动 fallback 到"沿主轴切片 → 每段凸包",
精度差一些但不需要新依赖。

用法:
    python scripts/decompose_vessel.py \
        --input projects/cache/VascuModel_collision.stl \
        --output-dir projects/cache/hulls \
        --backend auto --slices 12
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def _try_import_coacd():
    try:
        import coacd  # type: ignore
        return coacd
    except Exception:
        return None


def _try_import_pyvhacd():
    try:
        import pyvhacd  # type: ignore
        return pyvhacd
    except Exception:
        return None


def _decompose_coacd(mesh_path: Path, out_dir: Path, threshold: float) -> list[Path]:
    """V-HACD 风格的真正凸分解(质量高,需 pip install coacd)。"""
    import trimesh  # type: ignore
    coacd = _try_import_coacd()
    assert coacd is not None
    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh_obj = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(mesh_obj, threshold=threshold)
    paths = []
    for idx, (verts, faces) in enumerate(parts):
        out_path = out_dir / f"hull_{idx:03d}.stl"
        trimesh.Trimesh(vertices=verts, faces=faces).export(str(out_path))
        paths.append(out_path)
    return paths


def _decompose_axial_slices(mesh_path: Path, out_dir: Path, n_slices: int) -> list[Path]:
    """Fallback:沿主轴均匀切片,每段取凸包 — 仅用 PyVista/VTK。

    切完后再用连通分量分裂(避免分叉处单段一个大凸包套住空腔),
    每个连通片段取凸包写出。
    """
    import pyvista as pv

    mesh = pv.read(str(mesh_path))
    if mesh.n_cells == 0:
        raise RuntimeError(f"STL 是空的: {mesh_path}")

    # 找最长的主轴方向作切片轴
    b = np.asarray(mesh.bounds, dtype=float)
    extents = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    axis = int(np.argmax(extents))
    axis_name = "xyz"[axis]
    lo, hi = float(b[axis * 2]), float(b[axis * 2 + 1])

    n_slices = int(max(1, n_slices))
    edges = np.linspace(lo, hi, n_slices + 1)

    paths: list[Path] = []
    counter = 0
    for i in range(n_slices):
        # box-clip 出区间 [edges[i], edges[i+1]]
        bb = [-1e9, 1e9, -1e9, 1e9, -1e9, 1e9]
        bb[axis * 2] = edges[i]
        bb[axis * 2 + 1] = edges[i + 1]
        slab = mesh.clip_box(bb, invert=False, crinkle=False)
        try:
            slab = slab.extract_surface(algorithm=None).triangulate()
        except Exception:
            continue
        if slab.n_points < 4 or slab.n_cells == 0:
            continue
        # 连通分量分裂(每段沿轴切完可能有多个分支)
        comps = slab.connectivity(extraction_mode="all", largest=False)
        # PyVista connectivity 给每个点贴 RegionId,我们按 RegionId 取每个分量
        try:
            region_ids = np.asarray(comps.point_data.get("RegionId"))
        except Exception:
            region_ids = None
        if region_ids is None or region_ids.size == 0:
            # 没有 RegionId 就直接整段算
            unique_regions = [None]
        else:
            unique_regions = sorted(set(int(r) for r in region_ids.tolist()))
        for rid in unique_regions:
            if rid is None:
                part = comps
            else:
                part = comps.threshold(
                    value=(rid - 0.5, rid + 0.5),
                    scalars="RegionId",
                    preference="point",
                ).extract_surface(algorithm=None).triangulate()
            if part.n_points < 4 or part.n_cells == 0:
                continue
            hull = pv.PolyData(np.asarray(part.points, dtype=float)).delaunay_3d()
            try:
                hull_surf = hull.extract_surface(algorithm=None).triangulate()
            except Exception:
                hull_surf = hull
            if hull_surf.n_points < 4:
                continue
            out_path = out_dir / f"hull_{counter:03d}.stl"
            hull_surf.save(str(out_path))
            paths.append(out_path)
            counter += 1

    if not paths:
        # 完全降级:整段一个凸包
        hull = pv.PolyData(np.asarray(mesh.points, dtype=float)).delaunay_3d()
        hull_surf = hull.extract_surface(algorithm=None).triangulate()
        out_path = out_dir / "hull_000.stl"
        hull_surf.save(str(out_path))
        paths.append(out_path)
        print(f"[fallback] 轴向切片全空,退化为单个凸包: {out_path}")
    else:
        print(f"[axial-slices] 主轴={axis_name}, slices={n_slices} → {len(paths)} 个 hull")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="输入血管 STL(closed mesh 最好)")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="输出目录,生成 hull_NNN.stl + manifest.json")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "coacd", "axial"],
                    help="auto: 优先 coacd,否则 axial-slices")
    ap.add_argument("--slices", type=int, default=12,
                    help="axial 模式下的切片数")
    ap.add_argument("--coacd-threshold", type=float, default=0.05,
                    help="coacd 凹度阈值(越小越多 hull)")
    ap.add_argument("--clean", action="store_true",
                    help="生成前清空输出目录")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"输入文件不存在: {args.input}", file=sys.stderr)
        return 1

    out_dir: Path = args.output_dir
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = args.backend
    if backend == "auto":
        backend = "coacd" if _try_import_coacd() is not None else "axial"

    print(f"输入: {args.input}")
    print(f"输出: {out_dir}")
    print(f"后端: {backend}")

    if backend == "coacd":
        paths = _decompose_coacd(args.input, out_dir, args.coacd_threshold)
    elif backend == "axial":
        paths = _decompose_axial_slices(args.input, out_dir, args.slices)
    else:
        print(f"未知 backend: {backend}", file=sys.stderr)
        return 2

    manifest = {
        "source": str(args.input),
        "backend": backend,
        "n_hulls": len(paths),
        "hulls": [p.name for p in paths],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"完成,共 {len(paths)} 个 hull → {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
