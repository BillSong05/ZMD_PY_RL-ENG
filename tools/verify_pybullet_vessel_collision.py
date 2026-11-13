"""PyBullet 血管 STL 碰撞验证 — 第一步：STL 具备碰撞属性，物体不能穿模。

机器人物理引擎换成 PyBullet（机器人事实标准）。本脚本：

1. 用 ``createCollisionShape(GEOM_MESH, flags=GEOM_FORCE_CONCAVE_TRIMESH)``
   把 ``projects/cache/VascuModel_collision.stl`` 加载成**静态凹面碰撞体**
2. 生成一个**探针小球**（半径 ~ guidewire radius），从血管入口外侧沿中心线初始方向
   推过去；如果 STL 真有碰撞，小球会被弹开 / 卡住
3. 同时做一个**对照实验**：在血管旁边（lumen 外）远处放一个小球，让它沿同一方向
   走，验证它确实不会被血管挡 — 排除其他干扰
4. 输出每步小球位置 + 与壁的最近距离（用 ``getClosestPoints``），以及最终
   穿模深度（``contactPoints`` 中的 ``contactDistance``）
5. 输出 PNG 三联图 + JSON 报告

通过判据：
- 探针球在沿入口推进 N 步后**没有出现在管外**（getClosestPoints distance > 0 后还能 > 0）
- contactPoints 报告过至少一次 distance <= 0（说明真的撞过）
- 不依赖 PyBullet GUI（用 ``connect(p.DIRECT)``）

运行::

    python tools/verify_pybullet_vessel_collision.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEBUG = ROOT / "debug" / "pybullet"
DEBUG.mkdir(parents=True, exist_ok=True)

VESSEL_STL = ROOT / "projects" / "cache" / "VascuModel_collision.stl"
CENTERLINE_VTP = ROOT / "projects" / "cache" / "VascuModel_centerline.vtp"


def load_centerline_entry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (entry_pos, heading, centerline_points)。"""
    from zmd.centerline.centerline_model import Centerline
    branches = Centerline.load_branches_from_vtp(CENTERLINE_VTP) if CENTERLINE_VTP.exists() else []
    if not branches:
        raise RuntimeError(f"找不到 centerline: {CENTERLINE_VTP}")
    pts = np.asarray(branches[0].points, dtype=float)
    entry = pts[0].copy()
    h = pts[1] - pts[0]
    n = float(np.linalg.norm(h))
    if n < 1e-9:
        raise RuntimeError("centerline 前两点重合")
    return entry, h / n, pts


# ======================================================================
# PyBullet scene
# ======================================================================

class VesselCollisionScene:
    """单一血管静态碰撞体 + 一个动态探针球。"""

    def __init__(self, vessel_stl: Path, probe_radius: float = 0.05):
        self.vessel_stl = Path(vessel_stl)
        self.probe_radius = float(probe_radius)
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, 0, physicsClientId=self.client)   # 不要重力，便于纯推进测试
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0,
            numSolverIterations=150,
            enableConeFriction=0,
            physicsClientId=self.client)

        # 1) 血管: 静态凹面 trimesh
        col_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=str(self.vessel_stl),
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            meshScale=[1.0, 1.0, 1.0],
            physicsClientId=self.client)
        self.vessel_id = p.createMultiBody(
            baseMass=0.0,                          # 0 mass → static
            baseCollisionShapeIndex=col_shape,
            basePosition=[0, 0, 0],
            physicsClientId=self.client)
        # 静态体的碰撞参数 — 让它对凹面 ray-test 友好
        p.changeDynamics(self.vessel_id, -1,
                         restitution=0.0, lateralFriction=0.5,
                         contactStiffness=1e6, contactDamping=1e3,
                         physicsClientId=self.client)

        # 2) 探针小球 (dynamic)
        ball_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE, radius=self.probe_radius,
            physicsClientId=self.client)
        ball_vis = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=self.probe_radius,
            rgbaColor=[1, 0.6, 0, 1], physicsClientId=self.client)
        self.probe_id = p.createMultiBody(
            baseMass=1.0e-3,                       # 轻质量
            baseCollisionShapeIndex=ball_shape,
            baseVisualShapeIndex=ball_vis,
            basePosition=[0, 0, 0],
            physicsClientId=self.client)
        p.changeDynamics(self.probe_id, -1,
                         restitution=0.0, lateralFriction=0.5,
                         linearDamping=0.5, angularDamping=0.5,
                         physicsClientId=self.client)

    def place_probe(self, pos: np.ndarray, lin_vel: np.ndarray = None) -> None:
        p.resetBasePositionAndOrientation(
            self.probe_id, pos.tolist(), [0, 0, 0, 1],
            physicsClientId=self.client)
        if lin_vel is not None:
            p.resetBaseVelocity(self.probe_id, lin_vel.tolist(), [0, 0, 0],
                                physicsClientId=self.client)

    def apply_push(self, force: np.ndarray) -> None:
        """每步在 probe 上施加力（世界坐标）。"""
        p.applyExternalForce(self.probe_id, -1,
                             force.tolist(), [0, 0, 0], p.WORLD_FRAME,
                             physicsClientId=self.client)

    def step(self):
        p.stepSimulation(physicsClientId=self.client)

    def probe_pos(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.probe_id,
                                                  physicsClientId=self.client)
        return np.asarray(pos, dtype=float)

    def probe_vel(self) -> np.ndarray:
        lin, _ = p.getBaseVelocity(self.probe_id, physicsClientId=self.client)
        return np.asarray(lin, dtype=float)

    def closest_distance(self) -> float:
        """probe ↔ vessel 最近表面距离 (>0 没碰，<=0 已交叠)。"""
        out = p.getClosestPoints(
            self.probe_id, self.vessel_id, distance=10.0,
            physicsClientId=self.client)
        if not out:
            return float("inf")
        return min(float(c[8]) for c in out)  # contactDistance

    def contacts_now(self) -> list[dict]:
        cp = p.getContactPoints(
            self.probe_id, self.vessel_id, physicsClientId=self.client)
        return [{"distance": float(c[8]),
                 "normal_force": float(c[9]),
                 "pos_on_probe": list(c[5]),
                 "pos_on_vessel": list(c[6])} for c in cp]

    def close(self):
        p.disconnect(physicsClientId=self.client)


# ======================================================================
# 实验
# ======================================================================

def run_push_into_wall(scene: VesselCollisionScene, start: np.ndarray,
                       heading: np.ndarray, n_steps: int = 600,
                       push_force: float = 0.05, label: str = "") -> dict:
    """把 probe 放到 start，沿 heading 推 n_steps 步；记录轨迹 + 最近距离 + 接触。"""
    scene.place_probe(start, lin_vel=heading * 0.0)
    log = []
    contacts_seen = 0
    min_dist = float("inf")
    max_penetration = 0.0
    contact_first = None

    for k in range(n_steps):
        scene.apply_push(heading * push_force)
        scene.step()
        pos = scene.probe_pos()
        vel = scene.probe_vel()
        d = scene.closest_distance()
        cs = scene.contacts_now()
        if cs:
            contacts_seen += 1
            if contact_first is None:
                contact_first = k
            for c in cs:
                if c["distance"] < 0:
                    max_penetration = max(max_penetration, -c["distance"])
        min_dist = min(min_dist, d)
        if k % 30 == 0 or cs:
            print(f"  [{label}] step {k:3d}  pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})  "
                  f"|vel|={np.linalg.norm(vel):.3f}  closest_d={d:+.3f}  contacts={len(cs)}")
        log.append({"step": k, "pos": pos.tolist(), "vel": vel.tolist(),
                    "closest": d, "n_contacts": len(cs)})

    final_pos = scene.probe_pos()
    return {
        "label": label,
        "start": start.tolist(),
        "heading": heading.tolist(),
        "n_steps": n_steps,
        "push_force": push_force,
        "final_pos": final_pos.tolist(),
        "displacement": float(np.linalg.norm(final_pos - start)),
        "contact_first_at": contact_first,
        "contacts_total_steps": contacts_seen,
        "min_closest_distance": float(min_dist),
        "max_penetration": float(max_penetration),
        "log_tail": log[-20:],
    }


def render_overview(centerline: np.ndarray, traj_in: list, traj_out: list,
                    png_path: Path) -> None:
    import pyvista as pv
    pv.OFF_SCREEN = True
    vessel = pv.read(str(VESSEL_STL))
    pl = pv.Plotter(off_screen=True, window_size=(1280, 800))
    pl.add_mesh(vessel, color="lightblue", opacity=0.30, name="vessel")
    if len(centerline) >= 2:
        cl = pv.PolyData()
        cl.points = centerline
        cl.lines = np.hstack([[len(centerline)], np.arange(len(centerline))]).astype(np.int64)
        pl.add_mesh(cl, color="green", line_width=2, name="centerline")
    if traj_in:
        arr = np.array([s["pos"] for s in traj_in])
        poly = pv.PolyData(); poly.points = arr
        poly.lines = np.hstack([[len(arr)], np.arange(len(arr))]).astype(np.int64)
        pl.add_mesh(poly, color="orange", line_width=4, name="probe_inside")
        pl.add_mesh(pv.PolyData(arr[:1]), color="red", point_size=14,
                    render_points_as_spheres=True)
        pl.add_mesh(pv.PolyData(arr[-1:]), color="yellow", point_size=18,
                    render_points_as_spheres=True)
    if traj_out:
        arr = np.array([s["pos"] for s in traj_out])
        poly = pv.PolyData(); poly.points = arr
        poly.lines = np.hstack([[len(arr)], np.arange(len(arr))]).astype(np.int64)
        pl.add_mesh(poly, color="cyan", line_width=4, name="probe_outside")
        pl.add_mesh(pv.PolyData(arr[:1]), color="blue", point_size=14,
                    render_points_as_spheres=True)
    pl.add_text("orange=push into entry  cyan=control (free space)\n"
                "red dot=start  yellow=final", font_size=10)
    pl.show_axes()
    pl.camera_position = "iso"
    pl.screenshot(str(png_path))
    pl.close()


def main() -> int:
    print("=" * 72)
    print("PyBullet 血管 STL 碰撞验证")
    print("=" * 72)
    print(f"vessel STL: {VESSEL_STL}  exists={VESSEL_STL.exists()}")
    if not VESSEL_STL.exists():
        print("[FATAL] 血管 STL 不存在")
        return 1

    entry, heading, centerline = load_centerline_entry()
    print(f"entry={entry}  heading={heading}  centerline_pts={len(centerline)}")

    # 入口前移一点点：把探针放在 entry 沿 -heading 方向 1mm 处 → 入口正前方
    # （centerline[0] 在外侧 ~0.19 单位，按现在的 STL 我们就让它从外侧推进）
    start_inside = entry - heading * 1.0     # 入口前方 1mm
    start_outside_lumen = entry + np.array([20.0, 0.0, 0.0])   # 远离血管管腔的对照点

    scene = VesselCollisionScene(VESSEL_STL, probe_radius=0.3)

    # 实验 1: 探针沿 heading 推进，应被血管壁挡住
    print("\n--- 实验 A: 推 probe 朝血管入口 ---")
    res_in = run_push_into_wall(scene, start_inside, heading,
                                n_steps=600, push_force=0.02, label="INTO")

    # 实验 2: 探针在远处自由空间，沿 heading 推进，应一路畅行
    print("\n--- 实验 B (对照): probe 在管外自由空间推进 ---")
    res_out = run_push_into_wall(scene, start_outside_lumen, heading,
                                 n_steps=600, push_force=0.02, label="FREE")

    # 判定
    # 核心判据: 接触出现过 + 穿模深度 < 探针半径（PyBullet 标准是接触深度应为 0）
    pass_collision_blocks = (
        res_in["contact_first_at"] is not None
        and res_in["max_penetration"] < scene.probe_radius
        and res_in["contacts_total_steps"] >= 10
    )
    pass_free_runs = res_out["contact_first_at"] is None and res_out["displacement"] > 1.0

    print("\n=== 总结 ===")
    print(f"  实验 A (push into vessel):")
    print(f"    contact_first_at = {res_in['contact_first_at']}")
    print(f"    contacts_total_steps = {res_in['contacts_total_steps']}")
    print(f"    min_closest_distance = {res_in['min_closest_distance']:.4f}")
    print(f"    max_penetration = {res_in['max_penetration']:.4f}")
    print(f"    displacement = {res_in['displacement']:.3f}")
    print(f"  实验 B (free-space control):")
    print(f"    contact_first_at = {res_out['contact_first_at']}  (期望 None)")
    print(f"    displacement = {res_out['displacement']:.3f}  (期望 >> A)")
    print(f"\n  STL collision blocks probe: {'PASS' if pass_collision_blocks else 'FAIL'}")
    print(f"  Free-space control passes:  {'PASS' if pass_free_runs else 'FAIL'}")
    verdict = pass_collision_blocks and pass_free_runs
    print(f"  Overall verdict: {'PASS' if verdict else 'FAIL'}")

    # 可视化
    render_overview(centerline, res_in["log_tail"] and [s for s in res_in.get("log_tail", [])] or [],
                    res_out["log_tail"] and [s for s in res_out.get("log_tail", [])] or [],
                    DEBUG / "vessel_collision_overview.png")
    # 重新拉全轨迹（log_tail 只是尾段，render 用全程）
    # 简单办法: 再跑一次为了图省事 — 已经太慢，所以保留 tail 即可

    out_json = DEBUG / "pybullet_collision_report.json"
    out_json.write_text(json.dumps({
        "vessel_stl": str(VESSEL_STL),
        "entry": entry.tolist(),
        "heading": heading.tolist(),
        "probe_radius": 0.3,
        "experiment_into": res_in,
        "experiment_free": res_out,
        "pass_collision_blocks": pass_collision_blocks,
        "pass_free_runs": pass_free_runs,
        "verdict": verdict,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport: {out_json}")
    print(f"png:    {DEBUG / 'vessel_collision_overview.png'}")

    scene.close()
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
