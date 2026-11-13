"""MuJoCo capsule-chain 医疗导丝 backend(参考 cathsim/airvlab)。

主要特性:
- 段数 / 段长 / 半径 / 刚度衰减 / 阻尼 / 密度 / 摩擦 通过 ``MujocoGuidewireConfig`` 配置
- 每段两轴 hinge (X/Y),刚度沿链 ×decay 衰减(近端硬,远端软)
- 近端 ``prox_slide`` + ``prox_twist`` 双自由度,velocity + general 双 actuator
- 可选加载凸分解后的血管 hull 集(``projects/cache/hulls/``)做真实碰撞
- 可选 J-tip 预弯(末端 N 段绕 X 轴累计偏角)

contype / conaffinity 约定:
- guidewire seg: contype=2, conaffinity=4(段间不自碰)
- vessel hull:   contype=4, conaffinity=2(只跟导丝碰)
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from zmd.simulation.guidewire.centerline_tube_constraint import CenterlineTubeConstraint
from zmd.simulation.guidewire.mujoco_config import MujocoGuidewireConfig
from zmd.simulation.mujoco_loader import import_mujoco


# ZMD_GW_DEBUG=1 时,每次 substep blocked / wall-slide 失败打印完整 block_reason。
_DEBUG_STEP = os.environ.get("ZMD_GW_DEBUG", "") not in ("", "0", "false", "False")


def _quat_from_z_to_dir(direction: np.ndarray) -> np.ndarray:
    """返回把局部 +Z 旋到 ``direction`` 的 MuJoCo wxyz 四元数。"""
    z_axis = np.array([0.0, 0.0, 1.0])
    d = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])

    d = d / norm
    dot = float(np.dot(z_axis, d))
    if dot > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-9:
        return np.array([0.0, 1.0, 0.0, 0.0])

    axis = np.cross(z_axis, d)
    quat = np.array([1.0 + dot, axis[0], axis[1], axis[2]])
    return quat / float(np.linalg.norm(quat))


def _default_hulls_dir() -> Path:
    """约定:``projects/cache/hulls/`` 下放 hull_NNN.stl + manifest.json。"""
    here = Path(__file__).resolve()
    # zmd/simulation/backends/mujoco_guidewire_backend.py → project root
    return here.parents[3] / "projects" / "cache" / "hulls"


class MujocoGuidewireBackend:
    """配置驱动的 MuJoCo 医疗导丝。"""

    def __init__(
        self,
        entry_pos,
        entry_dir,
        config: Optional[MujocoGuidewireConfig] = None,
        centerline_points: Optional[np.ndarray] = None,
        lumen_radius=None,   # float 或 np.ndarray(per-point)
        # 兼容旧接口(workspace_page / 测试)— 任意一个传入都会落到 config 上
        num_segments: Optional[int] = None,
        segment_length: Optional[float] = None,
        radius: Optional[float] = None,
        action_substeps: Optional[int] = None,
        mj_substeps: Optional[int] = None,
        enable_contact: Optional[bool] = None,
        enable_wall_slide: Optional[bool] = None,
        **_unused,
    ):
        mujoco = import_mujoco(verify_core=True)

        self._mj = mujoco
        self.entry_pos = np.asarray(entry_pos, dtype=float).reshape(3).copy()

        direction = np.asarray(entry_dir, dtype=float).reshape(3)
        direction_norm = float(np.linalg.norm(direction))
        self.entry_dir = (
            direction / direction_norm
            if direction_norm > 1e-9
            else np.array([0.0, 0.0, 1.0])
        )

        cfg = config or MujocoGuidewireConfig()
        # 旧关键字参数 override(逐项)
        if num_segments is not None:
            cfg = cfg.copy(n_segments=int(num_segments))
        if segment_length is not None:
            cfg = cfg.copy(segment_length=float(segment_length))
        if radius is not None:
            cfg = cfg.copy(radius=float(radius))
        if action_substeps is not None:
            cfg = cfg.copy(action_substeps=int(action_substeps))
        if mj_substeps is not None:
            cfg = cfg.copy(mj_substeps_per_action=int(mj_substeps))
        if enable_contact is not None:
            cfg = cfg.copy(enable_collision=bool(enable_contact))
        # enable_wall_slide 是历史 MVP 概念,新版改用 CenterlineTubeConstraint
        del enable_wall_slide

        self._centerline_points = (
            np.asarray(centerline_points, dtype=float)
            if centerline_points is not None else None
        )
        self._lumen_radius = lumen_radius
        self._tube_constraint: Optional[CenterlineTubeConstraint] = None
        self._seg_body_ids: list[int] = []

        # SDF 碰撞：由外部(workspace_page)通过 set_collision_sdf 注入。
        # None = 无血管 SDF，所有步骤直接放行(不检查穿模)。
        self._sdf_fn = None
        # 最近一次 SDF 检查与滑动结果（调试用）
        self._last_min_sdf: float = float("nan")
        self._last_sdf_blocked: bool = False          # 兼容旧逻辑
        self._last_status: str = "ok"                 # "ok" / "wall_slide" / "sdf_blocked"
        self._last_wall_slide_attempted: bool = False
        self._last_wall_slide_success: bool = False
        self._last_wall_slide_reason: str = "none"    # 详细原因串
        self._last_slide_dir: Optional[np.ndarray] = None  # 上次成功的滑动方向（fallback）
        # 进度跟踪：tip 投影到中心线的弧长（单调），区别于 inserted_length
        self._last_progress_s: float = 0.0
        self._last_motion_forward: bool = True
        # J-tip 转向最近一次的有效权重（调试用）
        self._last_j_tip_weight: float = 0.0
        # 新弱引导(中心线只作判断参考)的最近一次状态 —— 供 debug 面板观察
        self._last_angle_to_centerline: float = 0.0
        self._last_raw_move_dir: Optional[np.ndarray] = None
        self._last_center_tangent: Optional[np.ndarray] = None
        self._last_move_dir: Optional[np.ndarray] = None
        self._last_slide_allowed: bool = False
        # 推进意图状态: "centerline_assist" / "raw_forward" / "wall_fallback" / "blocked"
        self._last_status_intent: str = "raw_forward"
        # 双阈值 wall-slide 决策(避免轻微偏航就 blocked):aligned / yaw_tolerable / blocked
        self._last_slide_reason: str = "n/a"
        self._last_wall_normal: Optional[np.ndarray] = None
        self._last_candidate_slide_dir: Optional[np.ndarray] = None
        self._last_projection_angle_deg: float = float("nan")
        # 小窗"局部跟随相机"的平滑状态(低通滤波,避免抖动跳动)
        self._inset_camera_pos_smooth: Optional[np.ndarray] = None
        self._inset_focal_smooth: Optional[np.ndarray] = None
        self._inset_forward_smooth: Optional[np.ndarray] = None

        self._build_from_config(cfg)

        self.last_status = "idle"
        self.last_contact_stats: dict = {}
        self.last_constraint_stats: dict = {
            "max_radial_error": 0.0,
            "num_projected_bodies": 0,
            "tip_centerline_s": 0.0,
        }
        self.reset()

    # ------------------------------------------------------------------ build

    def _build_from_config(self, cfg: MujocoGuidewireConfig) -> None:
        """从 config 构造一个新的 MuJoCo model/data,并缓存索引。"""
        self.config = cfg
        self.num_segments = int(max(2, cfg.n_segments))
        self.segment_length = float(max(1e-4, cfg.segment_length))
        self.radius = float(max(1e-4, cfg.radius))
        self._max_length = self.num_segments * self.segment_length
        # 推进总路程由"起点→终点"的中心线弧长决定,而不是导丝自身长度。
        # 没有中心线时(纯物理模式)退化为导丝长度。
        if self._centerline_points is not None and len(self._centerline_points) >= 2:
            seg = np.linalg.norm(np.diff(self._centerline_points, axis=0), axis=1)
            self._travel_length = float(np.sum(seg))
        else:
            self._travel_length = self._max_length
        # 实际可推进上限:导丝自身物理长度。
        # 不再夹到中心线弧长(min(max_length, travel_length)) —— J-tip steering
        # 会让 wire 实际 tip_path 长于中心线弧长(路径在腔内蜿蜒),用 travel_length
        # 当 qpos 上限会导致 tip_s 投影只到 ~92% 时 qpos 已饱和、再推无效("卡 92%" bug)。
        # 物理边界由 SDF 在 _advance_tip_path 里把关:wire 撞到血管末端壁就自然 blocked,
        # 不需要 qpos 级硬夹。
        self._max_inserted = self._max_length
        self.action_substeps = int(max(1, cfg.action_substeps))
        self.mj_substeps_per_action = int(max(0, cfg.mj_substeps_per_action))

        hull_paths: list[Path] = []
        if cfg.enable_collision:
            hulls_dir = cfg.hulls_dir or _default_hulls_dir()
            if hulls_dir.exists():
                hull_paths = sorted(hulls_dir.glob("hull_*.stl"))
        self._loaded_hull_count = len(hull_paths)
        # 兼容旧 UI/test 字段
        self.enable_contact = bool(hull_paths)
        self.enable_wall_slide = False

        self._xml = self._build_xml(hull_paths)
        # MuJoCo 通过 assets 字典支持把外部文件作为内存绑定;但 from_xml_string 也可以读
        # XML 里 file="<absolute>" 的资产 — 直接走绝对路径最稳。
        # 注意:段链是深层嵌套 body,MuJoCo XML 解析递归,>78 段会爆主线程默认栈。
        # 改在加大栈(64MB)的线程里编译,可安全支持 80+ 段。
        self._model = self._compile_model_large_stack(self._xml)
        self._data = self._mj.MjData(self._model)

        self._slide_qpos_idx = self._joint_qpos_addr("prox_slide")
        self._twist_qpos_idx = self._joint_qpos_addr("prox_twist")
        self._site_ids: list[int] = []
        self._seg_body_ids = []
        for idx in range(self.num_segments):
            site_id = self._mj.mj_name2id(
                self._model, self._mj.mjtObj.mjOBJ_SITE, f"site_end_{idx}"
            )
            if site_id >= 0:
                self._site_ids.append(int(site_id))
            bid = self._mj.mj_name2id(
                self._model, self._mj.mjtObj.mjOBJ_BODY, f"seg_{idx}"
            )
            if bid >= 0:
                self._seg_body_ids.append(int(bid))

        # 中心线管腔约束(只有提供中心线且 config 启用时构建)
        self._tube_constraint = None
        if cfg.enable_centerline_constraint and self._centerline_points is not None:
            if self._lumen_radius is None:
                # 没给 lumen_radius 时,默认一个保守值:导丝半径 + 1.5 mm 余量
                lumen = float(self.radius) + 1.5
            else:
                lumen = self._lumen_radius
            self._tube_constraint = CenterlineTubeConstraint(
                centerline_points=self._centerline_points,
                lumen_radius=lumen,
                guidewire_radius=self.radius,
                k_spring=cfg.constraint_k_spring,
                k_damp=cfg.constraint_k_damp,
                margin=cfg.constraint_margin,
                k_attract=cfg.constraint_k_attract,
            )

        # 中心线弧长缓存(弱方向引导用:最近点 / 切向查询,只读不吸附)
        if self._centerline_points is not None and len(self._centerline_points) >= 2:
            cl = self._centerline_points
            self._cl_seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
            self._cl_s = np.concatenate([[0.0], np.cumsum(self._cl_seg)])
            # 投影到中心线的向量化缓存(避免每步 Python 循环遍历所有段)
            self._cl_a = cl[:-1]                       # (M,3) 各段起点
            self._cl_ab = cl[1:] - cl[:-1]             # (M,3) 各段向量
            self._cl_denom = np.einsum("ij,ij->i", self._cl_ab, self._cl_ab)  # (M,)
            self._cl_denom = np.where(self._cl_denom < 1e-12, 1e-12, self._cl_denom)
        else:
            self._cl_s = None
            self._cl_a = None
            self._cl_ab = None
            self._cl_denom = None

        # 导丝实际走过的路径(第二步核心):导丝沿自己的轨迹回填,而不是中心线。
        # reset() 会把它初始化为 [entry_pos]。
        self._tip_path: list[np.ndarray] = [self.entry_pos.copy()]

    def _build_xml(self, hull_paths: list[Path]) -> str:
        cfg = self.config
        dx = self.segment_length
        radius = self.radius
        qw, qx, qy, qz = _quat_from_z_to_dir(self.entry_dir)
        ex, ey, ez = self.entry_pos

        # ---- 资产 ----
        mesh_assets = []
        vessel_geoms = []
        for idx, p in enumerate(hull_paths):
            name = f"vessel_hull_{idx:03d}"
            mesh_assets.append(
                f'<mesh name="{name}" file="{p.as_posix()}"/>'
            )
            vessel_geoms.append(
                f'<geom name="g_{name}" type="mesh" mesh="{name}" '
                f'class="vessel" />'
            )
        mesh_assets_xml = "\n    ".join(mesh_assets)
        vessel_geoms_xml = "\n      ".join(vessel_geoms)

        # ---- 导丝段链 ----
        # cathsim 风格:从末端往近端嵌套构造
        body_xml = ""
        # J-tip 预弯:在最后 j_tip_segments 段累计 quat 偏移
        j_tip_n = max(0, int(cfg.j_tip_segments))
        j_tip_rad = math.radians(float(cfg.j_tip_angle_deg))
        per_seg_jtip = (j_tip_rad / j_tip_n) if (j_tip_n > 0 and j_tip_rad != 0.0) else 0.0

        # 段索引 0 在近端(基座),N-1 在末端(tip)
        for idx in range(self.num_segments - 1, -1, -1):
            stiff_local = float(cfg.stiffness) * (float(cfg.stiffness_decay) ** idx)
            if idx == 0:
                joints = (
                    f'<joint name="prox_slide" type="slide" axis="0 0 1" '
                    f'range="-{self._max_length:.9f} 0" '
                    f'damping="{cfg.proximal_slide_damping}"/>\n'
                    f'<joint name="prox_twist" type="hinge" axis="0 0 1" '
                    f'damping="{cfg.proximal_twist_damping}"/>'
                )
                pose = (
                    f'pos="{ex:.9f} {ey:.9f} {ez:.9f}" '
                    f'quat="{qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f}"'
                )
            else:
                joints = (
                    f'<joint name="bend_x_{idx}" type="hinge" axis="1 0 0" '
                    f'stiffness="{stiff_local:.6f}" damping="{cfg.damping}" '
                    f'armature="{cfg.armature}" range="-1.5 1.5"/>\n'
                    f'<joint name="bend_y_{idx}" type="hinge" axis="0 1 0" '
                    f'stiffness="{stiff_local:.6f}" damping="{cfg.damping}" '
                    f'armature="{cfg.armature}" range="-1.5 1.5"/>'
                )
                # J-tip:仅在末端最后 j_tip_n 段加 quat 偏置(绕 X 轴)
                tip_dist_from_end = self.num_segments - 1 - idx
                if per_seg_jtip != 0.0 and tip_dist_from_end < j_tip_n:
                    half = per_seg_jtip * 0.5
                    qjw, qjx = math.cos(half), math.sin(half)
                    pose = (
                        f'pos="0 0 {dx:.9f}" '
                        f'quat="{qjw:.9f} {qjx:.9f} 0 0"'
                    )
                else:
                    pose = f'pos="0 0 {dx:.9f}"'

            # 接触参数走默认 class="guidewire_seg"
            geom = (
                f'<geom name="seg_{idx}" type="capsule" '
                f'fromto="0 0 0 0 0 {dx:.9f}" size="{radius:.9f}" '
                f'class="guidewire_seg"/>'
            )
            site = f'<site name="site_end_{idx}" pos="0 0 {dx:.9f}" size="0.05"/>'
            body_xml = (
                f'<body name="seg_{idx}" {pose}>\n'
                f'{joints}\n{geom}\n{site}\n{body_xml}\n</body>'
            )

        # ---- 完整 XML ----
        return f"""<mujoco model="medical_guidewire">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{cfg.timestep}" gravity="0 0 0" integrator="implicitfast"
          iterations="20" tolerance="1e-7"/>
  <default>
    <default class="guidewire_seg">
      <geom type="capsule"
            rgba="0.78 0.82 0.88 1"
            density="{cfg.density}"
            contype="2" conaffinity="4"
            condim="{cfg.contact_condim}"
            friction="{cfg.friction_slide} 0.005 0.0001"
            margin="{cfg.contact_margin}"/>
    </default>
    <default class="vessel">
      <geom type="mesh"
            rgba="0.85 0.4 0.4 0.25"
            contype="4" conaffinity="2"
            condim="{cfg.contact_condim}"
            friction="{cfg.friction_slide} 0.005 0.0001"
            margin="{cfg.contact_margin}"/>
    </default>
  </default>
  <asset>
    {mesh_assets_xml}
  </asset>
  <worldbody>
    <body name="vessel_static" pos="0 0 0">
      {vessel_geoms_xml}
    </body>
    {body_xml}
  </worldbody>
  <actuator>
    <velocity name="slider_act" joint="prox_slide" kv="5" ctrlrange="-1 1"/>
    <motor name="rotator_act" joint="prox_twist" gear="1" ctrlrange="-1 1"/>
  </actuator>
</mujoco>"""

    def _compile_model_large_stack(self, xml: str):
        """在加大栈的线程里编译 MuJoCo 模型,避免深层嵌套 body 爆栈。

        段链是 N 层嵌套 body,MuJoCo XML 解析递归,>78 段会撑爆主线程默认栈
        (Windows 上表现为 STATUS_STACK_OVERFLOW / 0xC00000FD 硬崩溃)。
        在 64MB 栈的工作线程里编译可安全支持 80+ 段;段数少时该线程开销可忽略。
        编译失败/异常会在主线程重新抛出。
        """
        import threading

        holder: dict = {}

        def _work():
            try:
                holder["model"] = self._mj.MjModel.from_xml_string(xml)
            except BaseException as exc:  # noqa: BLE001 - 转抛回主线程
                holder["error"] = exc

        prev_size = threading.stack_size()
        try:
            threading.stack_size(64 * 1024 * 1024)  # 64 MB,足够 80+ 段
            t = threading.Thread(target=_work, name="mj-compile")
            t.start()
            t.join()
        finally:
            # 还原全局线程栈设置,不影响其它线程
            try:
                threading.stack_size(prev_size)
            except (ValueError, RuntimeError):
                pass

        if "error" in holder:
            raise holder["error"]
        return holder["model"]

    # --------------------------------------------------------- rebuild / API

    def rebuild(self, config: MujocoGuidewireConfig) -> None:
        """根据新 config 重建 model,尽量保留 inserted_length / rotation_angle。"""
        prev_inserted = self.inserted_length
        prev_rotation = self.rotation_angle
        self._build_from_config(config)
        self.reset()
        # 把进度按比例映射回去(总长可能变了)
        if prev_inserted > 0.0 and self._max_inserted > 0.0:
            new_len = float(np.clip(prev_inserted, 0.0, self._max_inserted))
            if self._slide_qpos_idx >= 0:
                self._data.qpos[self._slide_qpos_idx] = -self._max_length + new_len
        if self._twist_qpos_idx >= 0:
            self._data.qpos[self._twist_qpos_idx] = float(prev_rotation)
        self._mj.mj_forward(self._model, self._data)
        # 重建后按当前 inserted_length 沿弱引导重新生成路径(段数可能已变)
        if self._kinematic_mode_active():
            self._init_tip_path()
            self._update_motion_debug()

    def _preshape_to_centerline(self) -> None:
        """[已弃用 — 第二步移除] 旧的"强中心线贴合"逻辑:把 tip 直接放到 centerline(s),
        整根导丝按中心线摆位。现在导丝改为沿自身路径(_tip_path)运动,中心线只做弱方向
        引导,本方法已不在任何主流程(reset / _apply_action / step_simulation / rebuild)
        中调用,保留仅供历史参考,请勿重新接入。

        关键改造 (vs 旧切线累积版):
        1. 从 tip 出发,沿中心线 chord-walk(二分法)往回逐段找 site_end_i,
           使相邻两点的世界距离 == segment_length(导丝段是刚体,必须 chord = dx)。
           这避免了大曲率处"切线累积"产生的位置漂移。
        2. 各段方向 = (target_i - target_(i-1)) / dx,完全由位置决定,
           不再依赖切线插值。
        3. 储存阶段(s ≤ 0)沿 -entry_dir 平直延伸。
        """
        if self._centerline_points is None or self.num_segments < 2:
            return
        inserted = self.inserted_length
        dx = self.segment_length
        N = self.num_segments
        cl = self._centerline_points
        seg_lens = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        cl_len = float(s_cum[-1])
        entry_pos = self.entry_pos
        entry_dir = self.entry_dir

        def pos_at(s: float) -> np.ndarray:
            """弧长 s 处的中心线位置。s<0 沿 -entry_dir 外推,s>cl_len 沿末段切线外推。"""
            if s <= 0.0:
                return entry_pos + s * entry_dir
            if s >= cl_len:
                tail = cl[-1] - cl[-2]
                n = float(np.linalg.norm(tail))
                tail = tail / n if n > 1e-9 else entry_dir
                return cl[-1] + (s - cl_len) * tail
            idx = int(np.searchsorted(s_cum, s))
            idx = max(1, min(idx, len(cl) - 1))
            s0, s1 = s_cum[idx - 1], s_cum[idx]
            t = (s - s0) / max(s1 - s0, 1e-9)
            return cl[idx - 1] + t * (cl[idx] - cl[idx - 1])

        if inserted <= 0:
            # 整链在储存段 — hinge 全 0
            for i in range(1, N):
                ax = self._joint_qpos_addr(f"bend_x_{i}")
                ay = self._joint_qpos_addr(f"bend_y_{i}")
                if ax >= 0:
                    self._data.qpos[ax] = 0.0
                if ay >= 0:
                    self._data.qpos[ay] = 0.0
            return

        targets = np.zeros((N, 3), dtype=float)

        # ---- 分两阶段填 targets ----
        # Stage 1: 储存段(s_i ≤ 0)— 沿 -entry_dir 直线排列
        # Stage 2: 已出 entry 段(s_i > 0)— 沿 cl 做 chord=dx 向前 walk
        first_deployed = None
        for i in range(N):
            s_i = inserted - (N - 1 - i) * dx
            if s_i <= 0.0:
                targets[i] = entry_pos + s_i * entry_dir
            else:
                first_deployed = i
                break

        if first_deployed is not None:
            # 起点:若 first_deployed > 0,前一个 storage target;否则 entry_pos
            if first_deployed == 0:
                cur_pos = entry_pos
            else:
                cur_pos = targets[first_deployed - 1]
            cur_s = 0.0   # cl 上的弧长进度
            ds_step = 0.05

            def chord_walk_forward(cur_pos, cur_s, chord):
                """从 (cur_pos, cur_s) 出发,在 cl 上找新 s_new 使 |pos_at(s_new)-cur_pos| = chord。

                假设 cur_pos 离 cl 不远(在 entry 附近或在 cl 上);从 cur_s 单调向前
                以 ds_step 步长 walk,跨过 chord 时二分精修。
                走完 cl 沿末段切线外推。
                """
                s_a = cur_s
                p_a = pos_at(s_a) if s_a > 1e-9 else entry_pos
                d_a = float(np.linalg.norm(p_a - cur_pos))
                if d_a >= chord:
                    # cur_pos 已在 cl(s) 外的 chord 之外 — 极少见
                    # 退化:沿 (p_a - cur_pos) 方向走 chord
                    dirv = (p_a - cur_pos)
                    n = float(np.linalg.norm(dirv))
                    dirv = dirv / n if n > 1e-9 else entry_dir
                    return cur_pos + chord * dirv, cur_s
                # 单调向前
                s_b = s_a
                while True:
                    s_b = s_a + ds_step
                    if s_b >= cl_len:
                        s_b = cl_len
                        p_b = pos_at(s_b)
                        d_b = float(np.linalg.norm(p_b - cur_pos))
                        if d_b >= chord:
                            # cl 末段内已经跨过
                            lo, hi = s_a, s_b
                            for _ in range(30):
                                mid = 0.5 * (lo + hi)
                                if float(np.linalg.norm(pos_at(mid) - cur_pos)) < chord:
                                    lo = mid
                                else:
                                    hi = mid
                            s_found = 0.5 * (lo + hi)
                            return pos_at(s_found), s_found
                        # 还没跨 → tail 外推
                        tail = cl[-1] - cl[-2]
                        tn = float(np.linalg.norm(tail))
                        tail_dir = tail / tn if tn > 1e-9 else entry_dir
                        # 解 |cl_end + t*tail_dir - cur_pos| = chord
                        base = cl[-1] - cur_pos
                        bb = float(np.dot(base, tail_dir))
                        cc = float(np.dot(base, base)) - chord * chord
                        disc = bb * bb - cc
                        if disc < 0:
                            return cur_pos + chord * tail_dir, cl_len
                        t = -bb + float(np.sqrt(disc))
                        return cl[-1] + t * tail_dir, cl_len + t
                    p_b = pos_at(s_b)
                    d_b = float(np.linalg.norm(p_b - cur_pos))
                    if d_b >= chord:
                        # 二分 [s_a, s_b]
                        lo, hi = s_a, s_b
                        for _ in range(30):
                            mid = 0.5 * (lo + hi)
                            if float(np.linalg.norm(pos_at(mid) - cur_pos)) < chord:
                                lo = mid
                            else:
                                hi = mid
                        s_found = 0.5 * (lo + hi)
                        return pos_at(s_found), s_found
                    s_a = s_b
                    d_a = d_b

            for i in range(first_deployed, N):
                cur_pos, cur_s = chord_walk_forward(cur_pos, cur_s, dx)
                targets[i] = cur_pos

        self._last_preshape_targets = targets.copy()  # debug

        # 3) 跟踪每段累积世界旋转矩阵 R_chain[i] (列向量 = body i 的 ex/ey/ez 世界基)
        # seg_0 的世界 quat = entry quat(把 (0,0,1) 旋到 entry_dir),从中提取 R_0
        qw, qx, qy, qz = _quat_from_z_to_dir(entry_dir)
        R_chain = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ])
        for i in range(1, N):
            d = targets[i] - targets[i - 1]
            n = float(np.linalg.norm(d))
            t_world = d / n if n > 1e-9 else R_chain[:, 2]
            # parent 局部正交基:R_chain 列 = [ex, ey, ez]
            ex = R_chain[:, 0]
            ey = R_chain[:, 1]
            ez = R_chain[:, 2]
            tlx = float(np.dot(t_world, ex))
            tly = float(np.dot(t_world, ey))
            tlz = float(np.dot(t_world, ez))
            # XML joint 顺序 bend_x 先 bend_y 后(intrinsic):
            #   局部 (0,0,1) 经 R_x(θx) @ R_y(θy) → (sin θy, -sin θx cos θy, cos θx cos θy)
            theta_y = float(np.arcsin(float(np.clip(tlx, -0.999, 0.999))))
            theta_x = float(np.arctan2(-tly, tlz))
            ax_addr = self._joint_qpos_addr(f"bend_x_{i}")
            ay_addr = self._joint_qpos_addr(f"bend_y_{i}")
            if ax_addr >= 0:
                self._data.qpos[ax_addr] = theta_x
            if ay_addr >= 0:
                self._data.qpos[ay_addr] = theta_y
            # 更新 R_chain[i] = R_chain[i-1] @ R_x(θx) @ R_y(θy)
            cx, sx = float(np.cos(theta_x)), float(np.sin(theta_x))
            cy, sy = float(np.cos(theta_y)), float(np.sin(theta_y))
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            R_chain = R_chain @ Rx @ Ry

    # ------------------------------------------------------------------
    # 第二步:弱中心线方向引导 + 路径驱动(中心线只当导航参考,不当轨道)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # SDF 碰撞硬安全检查(防穿模)
    # ------------------------------------------------------------------

    def set_collision_sdf(self, sdf_fn) -> None:
        """注入血管有向距离场函数。

        sdf_fn(pos) -> float:
            负值 = 腔内(安全侧),正值 = 腔外(穿模侧)。
            由 vtkImplicitPolyDataDistance.EvaluateFunction 包装而来。
        传入 None 可关闭 SDF 检查。
        """
        self._sdf_fn = sdf_fn

    def _is_tip_safe(self, candidate: np.ndarray) -> bool:
        """检查候选 tip 点是否在血管腔内,不穿模。

        安全条件:sd <= -(guidewire_radius + collision_margin)
        即球心距管壁 >= guidewire_radius + collision_margin(Minkowski 膨胀)。
        无 SDF 时始终返回 True(放行)。
        异常时也返回 True(不因 SDF 错误阻断仿真)。
        """
        if self._sdf_fn is None:
            self._last_min_sdf = float("nan")
            return True
        threshold = -(float(self.radius) + float(self.config.collision_margin))
        try:
            sd = float(self._sdf_fn(candidate))
            self._last_min_sdf = sd
            return sd <= threshold
        except Exception:
            self._last_min_sdf = float("nan")
            return True  # SDF 调用异常 → 放行,不中断仿真

    def _is_candidate_wire_safe(self, candidate_tip: np.ndarray) -> bool:
        """检查"从当前 tip 推进到 candidate_tip 所新增的那一段"是否穿壁。

        修复:旧实现每步都重新采样过去 12mm 的历史路径(8 个回看点 × 多个插值)。
        历史点早已被验证安全 + SDF 是静态场,但 vtkImplicitPolyDataDistance 在
        边界附近有 ~1e-3 量级数值噪声,边缘老点偶发判 unsafe → 整步 blocked。
        新逻辑只检查"new 段":
          1. candidate_tip 本身(tip 安全检查 + 整段半径膨胀);
          2. 从 prev_tip 到 candidate_tip 之间均匀插值 (sdf_check_segment_samples) 个点
             — 防"两端安全但中间直线穿壁"(只有当 prev→cand 跨度较大时才需要);
          3. 紧邻 prev_tip 的 1 个回看点(catches "推进端在 cand 处安全,但 cand 引出
             的尾部翘起一点穿壁"的极端情况,但只用 1 个,不用 8 个老点)。

        无 SDF 时放行;异常时保守返回 False。
        """
        if self._sdf_fn is None:
            return True
        try:
            cand = np.asarray(candidate_tip, dtype=float).reshape(3)
            threshold = -(float(self.radius) + float(self.config.collision_margin))
            min_sdf = float("inf")

            samples: list[np.ndarray] = [cand]
            if len(self._tip_path) >= 1:
                prev_tip = np.asarray(self._tip_path[-1], dtype=float)
                seg_samples = int(max(0, self.config.sdf_check_segment_samples))
                # 仅在新增段非零长时插值
                seg_len = float(np.linalg.norm(cand - prev_tip))
                if seg_len > 1e-9 and seg_samples > 0:
                    for k in range(1, seg_samples + 1):
                        t = k / float(seg_samples + 1)
                        samples.append((1.0 - t) * prev_tip + t * cand)
                # 仅 1 个回看点(prev_tip 自身),足够 catch 尾部翘起穿壁。
                # 历史更远的点不再重采,避免 SDF 数值噪声误杀。
                samples.append(prev_tip)

            for p in samples:
                try:
                    sd = float(self._sdf_fn(p))
                except Exception:
                    continue
                if not np.isfinite(sd):
                    continue
                min_sdf = min(min_sdf, sd)
                if sd > threshold:
                    self._last_min_sdf = min_sdf
                    return False
            if np.isfinite(min_sdf):
                self._last_min_sdf = min_sdf
            return True
        except Exception:
            self._last_status = "sdf_check_error"
            return False

    def _find_contact_point(
        self, p_safe: np.ndarray, p_unsafe: np.ndarray
    ) -> np.ndarray:
        """在安全点和不安全点之间二分，找到最靠近壁面的最后安全点。

        p_safe  — 当前已知安全的 tip 位置。
        p_unsafe — 被 _is_tip_safe 判定为不安全的 candidate。
        返回约 8 步二分后靠近壁面的安全边界点（用于估计法向和出发滑动）。
        """
        lo = np.asarray(p_safe, dtype=float).copy()
        hi = np.asarray(p_unsafe, dtype=float).copy()
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            if self._is_tip_safe(mid):
                lo = mid
            else:
                hi = mid
        return lo

    def _estimate_sdf_normal(self, p: np.ndarray) -> Optional[np.ndarray]:
        """有限差分估计 p 点处的 SDF 梯度方向(指向腔外)。

        梯度 = ∇SDF,方向从 sd 小处(腔内)指向 sd 大处(腔外)。
        梯度太小或包含 NaN 时返回 None。
        """
        if self._sdf_fn is None:
            return None
        eps = float(self.config.sdf_normal_eps)
        try:
            p = np.asarray(p, dtype=float)
            e = np.eye(3) * eps
            dx = float(self._sdf_fn(p + e[0])) - float(self._sdf_fn(p - e[0]))
            dy = float(self._sdf_fn(p + e[1])) - float(self._sdf_fn(p - e[1]))
            dz = float(self._sdf_fn(p + e[2])) - float(self._sdf_fn(p - e[2]))
            g = np.array([dx, dy, dz]) / (2.0 * eps)
            if not np.all(np.isfinite(g)):
                return None
            n = float(np.linalg.norm(g))
            if n < 1e-9:
                return None
            return g / n
        except Exception:
            return None

    def _project_to_wall_tangent(
        self, move_dir: np.ndarray, normal: np.ndarray
    ) -> Optional[np.ndarray]:
        """把 move_dir 投影到壁面切平面(去掉沿法向的分量)。

        tangent = move_dir - dot(move_dir, normal) * normal
        切向太小(运动方向几乎完全垂直壁面)时返回 None。
        """
        tangent = move_dir - float(np.dot(move_dir, normal)) * normal
        n = float(np.linalg.norm(tangent))
        if n < 1e-9:
            return None
        return tangent / n

    def _try_wall_slide(
        self, tip_pos: np.ndarray, move_dir: np.ndarray, step_size: float,
        *,
        slide_dir: Optional[np.ndarray] = None,
        contact: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
        raw_move_dir: Optional[np.ndarray] = None,
        allow_center_blend: bool = False,
    ) -> bool:
        """碰壁后沿壁面切向滑动 — **多候选搜索**(不再"一次失败就 blocked")。

        修复点:
          - 旧实现只试一个 slide_dir,只缩半步长;切向估错就死。
          - 新实现枚举 5 类候选方向 × 4 档步长,取首个安全且 dot(dir, raw)>0 的;
          - inward_bias 由 `wall_slide_inward_bias_factor × radius` 决定(显著大于
            SDF 数值噪声),避免候选点贴壁时被边界判 unsafe。
          - allow_center_blend=True 时(仅 aligned/yaw_tolerable 区间),候选里加入
            中心线切向混合方向 — 这不是牵引,只是"提供一个稳定的备选切向"。
        中心线方向**永远不**作为主方向源(避免自动导航)。

        kwargs:
          slide_dir   — caller 算好的主方向(project(raw_move_dir, wall plane));
                        本函数把它作为 candidate[0]。
          raw_move_dir — 用于反向过滤 dot(cand, raw)>0;不给就退回 move_dir。
        """
        if self._sdf_fn is None:
            return False

        self._last_wall_slide_attempted = True
        self._last_wall_slide_success = False

        tip_pos = np.asarray(tip_pos, dtype=float)
        move_norm = float(np.linalg.norm(move_dir))
        if not np.isfinite(move_norm) or move_norm < 1e-9:
            self._last_wall_slide_reason = "bad_move_dir"
            return False
        move_dir_u = np.asarray(move_dir, dtype=float) / move_norm
        raw_u = (np.asarray(raw_move_dir, dtype=float)
                 if raw_move_dir is not None else move_dir_u)
        rn = float(np.linalg.norm(raw_u))
        raw_u = raw_u / rn if rn > 1e-9 else move_dir_u

        # ---- 1. contact / normal:调用方算好就直接用,否则自行计算 ----
        if contact is None:
            blocked_candidate = tip_pos + move_dir_u * float(step_size)
            contact = self._find_contact_point(tip_pos, blocked_candidate)
        if normal is None:
            normal = self._estimate_sdf_normal(contact)
            if normal is None:
                self._last_wall_slide_reason = "no_normal"
                return False

        # ---- 2. 主 slide_dir(投影到壁面)----
        primary = slide_dir
        if primary is None:
            primary = self._project_to_wall_tangent(move_dir_u, normal)

        # ---- 3. 构造多候选方向(标签 + 单位向量)----
        candidates: list[tuple[str, np.ndarray]] = []
        if primary is not None:
            candidates.append(("primary", np.asarray(primary, dtype=float).copy()))
            # 与上次成功方向轻混(0.85/0.15),稳定性强
            if self._last_slide_dir is not None:
                mix = 0.85 * primary + 0.15 * self._last_slide_dir
                mn = float(np.linalg.norm(mix))
                if mn > 1e-9:
                    candidates.append(("mix_last", mix / mn))
            # aligned/yaw_tolerable 时允许混入中心线切向(不牵引,仅备选)
            if allow_center_blend:
                ct = self._nearest_centerline_tangent(contact)
                ctn = float(np.linalg.norm(ct))
                if ctn > 1e-9:
                    ct_u = ct / ctn
                    # 防反向:若中心线切向与 raw 反向,翻转一次(local 切向单边性)
                    if float(np.dot(ct_u, raw_u)) < 0.0:
                        ct_u = -ct_u
                    mix2 = 0.70 * primary + 0.30 * ct_u
                    m2n = float(np.linalg.norm(mix2))
                    if m2n > 1e-9:
                        candidates.append(("mix_center", mix2 / m2n))
            # 绕 wall_normal 左右微扰
            for ang_deg in (10.0, -10.0, 20.0, -20.0):
                rot = self._rotate_vector_around_axis(
                    primary, normal, math.radians(ang_deg)
                )
                rn2 = float(np.linalg.norm(rot))
                if rn2 > 1e-9:
                    candidates.append((f"rot_{int(ang_deg):+d}", rot / rn2))
        # 投影完全失败(head-on)时也试 last_slide_dir 投影
        if (not candidates) and self._last_slide_dir is not None:
            ls = self._project_to_wall_tangent(self._last_slide_dir, normal)
            if ls is not None:
                candidates.append(("last_slide_dir", ls))

        if not candidates:
            self._last_wall_slide_reason = "no_candidates"
            return False

        # ---- 4. 内推余量(显著大于 SDF 数值噪声,避免候选贴壁判 unsafe)----
        inward_dist = max(
            float(self.config.collision_margin) * 2.0,
            float(self.radius) * float(self.config.wall_slide_inward_bias_factor),
        )
        inward_bias = -normal * inward_dist

        base_step = float(step_size) * float(self.config.wall_slide_step_scale)
        # 4 档步长(从大到小),给方向更多挣扎空间
        step_scales = (1.0, 0.75, 0.5, 0.25)

        # ---- 5. 取第一个安全且不反向的候选 ----
        for label, sd in candidates:
            # 反向过滤:候选方向必须与 raw_move_dir 同向(dot > 0)
            if float(np.dot(sd, raw_u)) <= 0.0:
                continue
            for s in step_scales:
                cand_pos = contact + sd * (base_step * s) + inward_bias
                if self._is_candidate_wire_safe(cand_pos):
                    self._append_tip_path_point(cand_pos)
                    self._last_slide_dir = sd.copy()
                    self._last_wall_slide_success = True
                    self._last_wall_slide_reason = (
                        f"{label}@{s:.2f}"
                    )
                    self._last_status = "wall_slide"
                    return True

        self._last_wall_slide_reason = "candidates_unsafe"
        return False

    def _nearest_centerline_info(self, point: np.ndarray):
        """返回离 point 最近中心线点的 (距离, 切向单位向量)。无中心线时距离=0,切向=entry_dir。"""
        if self._cl_s is None:
            return 0.0, self.entry_dir.copy()
        cl = self._centerline_points
        d2 = np.sum((cl - point[None, :]) ** 2, axis=1)
        idx = int(np.argmin(d2))
        dist = float(np.sqrt(d2[idx]))
        # 切向:用相邻点差分
        lo = max(0, idx - 1)
        hi = min(len(cl) - 1, idx + 1)
        tang = cl[hi] - cl[lo]
        n = float(np.linalg.norm(tang))
        if n < 1e-9:
            return dist, self.entry_dir.copy()
        return dist, tang / n

    def _nearest_centerline_tangent(self, point: np.ndarray) -> np.ndarray:
        return self._nearest_centerline_info(point)[1]

    def _distance_to_centerline(self, point: np.ndarray) -> float:
        return self._nearest_centerline_info(point)[0]

    def _project_point_to_centerline_s(self, p: np.ndarray) -> float:
        """把点 p 投影到中心线(投影到线段,不是最近端点),返回对应弧长 s。

        用于任务进度:弱中心线模式下导丝会偏离中心线,inserted_length(走过的
        实际路径长度)不再等于"tip 在中心线上的弧长位置"。进度应基于后者。
        无中心线时退化为 0。
        """
        if self._cl_s is None or self._cl_ab is None:
            return 0.0
        p = np.asarray(p, dtype=float)
        # 向量化:一次性对所有段算投影参数 t、投影点 q、距离平方 d2
        ap = p[None, :] - self._cl_a                       # (M,3)
        t = np.einsum("ij,ij->i", ap, self._cl_ab) / self._cl_denom  # (M,)
        t = np.clip(t, 0.0, 1.0)
        q = self._cl_a + t[:, None] * self._cl_ab          # (M,3)
        diff = p[None, :] - q
        d2 = np.einsum("ij,ij->i", diff, diff)             # (M,)
        i = int(np.argmin(d2))
        best_s = float(self._cl_s[i] + t[i] * self._cl_seg[i])
        return float(min(max(best_s, 0.0), self._travel_length))

    def _centerline_point_at_s(self, s: float) -> Optional[np.ndarray]:
        """中心线弧长 s 处的点(沿折线线性插值);无中心线返回 None。

        用于内窥镜小窗第三人称相机:沿中心线往回退 back、往前看 fdist 都用它取点,
        使相机/焦点始终落在血管腔中轴上(跟随血管弯曲),而不是沿 tip 直线切向外飞。
        """
        if self._cl_s is None or self._cl_seg is None:
            return None
        total = float(self._cl_s[-1])
        s = float(min(max(s, 0.0), total))
        i = int(np.searchsorted(self._cl_s, s, side="right") - 1)
        i = max(0, min(i, len(self._cl_seg) - 1))
        seg = float(self._cl_seg[i])
        t = 0.0 if seg < 1e-12 else (s - float(self._cl_s[i])) / seg
        t = min(max(t, 0.0), 1.0)
        return self._cl_a[i] + t * self._cl_ab[i]

    def _lumen_radius_scalar(self, point: np.ndarray) -> float:
        """point 处的局部管腔半径(mm)。无数据时退回 导丝半径 + 1.5 余量。"""
        lr = self._lumen_radius
        if lr is None:
            return float(self.radius) + 1.5
        arr = np.asarray(lr, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        if (self._centerline_points is not None
                and len(arr) == len(self._centerline_points)):
            d2 = np.sum((self._centerline_points - np.asarray(point)[None, :]) ** 2, axis=1)
            return float(arr[int(np.argmin(d2))])
        return float(np.mean(arr))

    def tip_centerline_s(self) -> float:
        """当前 tip 在中心线上的投影弧长(单调,供 UI 进度用)。

        forward 时单调不减(避免分叉/噪声导致进度回跳);
        backward 时允许下降(重新按 tip 投影)。
        """
        tip = self.get_tip_position()
        s_raw = self._project_point_to_centerline_s(tip)
        if self._last_motion_forward:
            self._last_progress_s = max(self._last_progress_s, s_raw)
        else:
            self._last_progress_s = s_raw
        return float(self._last_progress_s)

    def _estimate_tip_direction(self) -> np.ndarray:
        """用导丝实际路径末端估计头部方向(回看一小段使方向平滑)。不足时退回 entry_dir。"""
        path = self._tip_path
        if len(path) < 2:
            return self.entry_dir.copy()
        tip = path[-1]
        # 从末端往回累积 ~一段长度,取回看点,避免相邻点过近导致方向噪声
        back_target = max(self.segment_length, 1.0)
        acc = 0.0
        ref = path[0]
        for j in range(len(path) - 1, 0, -1):
            acc += float(np.linalg.norm(path[j] - path[j - 1]))
            ref = path[j - 1]
            if acc >= back_target:
                break
        v = tip - ref
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return self.entry_dir.copy()
        return v / n

    # ------------------------------------------------------------------
    # J-tip 转向:twist 旋转 → J-tip 世界偏转方向 → 参与 move_dir
    # ------------------------------------------------------------------

    def _get_twist_angle(self) -> float:
        """返回 prox_twist 当前 qpos(弧度)。无 twist joint 时返回 0。"""
        if self._twist_qpos_idx < 0:
            return 0.0
        try:
            return float(self._data.qpos[self._twist_qpos_idx])
        except Exception:
            return 0.0

    @staticmethod
    def _rotate_vector_around_axis(v, axis, angle: float) -> np.ndarray:
        """Rodrigues 公式:把 v 绕 axis 旋转 angle 弧度。axis 退化时原样返回 v。"""
        v = np.asarray(v, dtype=float)
        axis = np.asarray(axis, dtype=float)
        an = float(np.linalg.norm(axis))
        if not np.isfinite(an) or an < 1e-8:
            return v
        k = axis / an
        ca, sa = float(np.cos(angle)), float(np.sin(angle))
        return v * ca + np.cross(k, v) * sa + k * float(np.dot(k, v)) * (1.0 - ca)

    def _estimate_j_tip_direction(self, shaft_dir: np.ndarray):
        """计算 J-tip 在世界坐标下的偏转方向(随 twist 绕导丝轴旋转)。

        - 取一个与 shaft_dir 垂直的参考方向 ref_dir(twist=0 时的偏转方向);
        - 把 ref_dir 绕 shaft_dir 旋转 twist_angle 得到世界偏转方向。
        J-tip 角度太小(<j_tip_min_angle_deg)时返回 None(直丝不转向)。
        """
        angle_deg = abs(float(getattr(self.config, "j_tip_angle_deg", 0.0)))
        if angle_deg < float(self.config.j_tip_min_angle_deg):
            return None

        shaft = np.asarray(shaft_dir, dtype=float)
        sn = float(np.linalg.norm(shaft))
        if not np.isfinite(sn) or sn < 1e-9:
            return None
        shaft = shaft / sn

        # 与 shaft 垂直的参考偏转方向(世界 Z 参考,退化用世界 Y)
        world_up = np.array([0.0, 0.0, 1.0])
        ref = np.cross(shaft, world_up)
        if float(np.linalg.norm(ref)) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            ref = np.cross(shaft, world_up)
        rn = float(np.linalg.norm(ref))
        if rn < 1e-9:
            return None
        ref = ref / rn

        # twist 绕导丝轴旋转参考方向
        twist = self._get_twist_angle()
        bend = self._rotate_vector_around_axis(ref, shaft, twist)
        bn = float(np.linalg.norm(bend))
        return bend / bn if bn > 1e-9 else None

    def _j_tip_effective_weight(self) -> float:
        """J-tip 转向的有效权重 = steering_weight * sin(j_tip_angle)。角度小则≈0。"""
        if not self.config.j_tip_steering_enabled:
            return 0.0
        angle_deg = abs(float(getattr(self.config, "j_tip_angle_deg", 0.0)))
        if angle_deg < float(self.config.j_tip_min_angle_deg):
            return 0.0
        return float(self.config.j_tip_steering_weight) * float(
            np.sin(np.deg2rad(angle_deg))
        )

    def _append_tip_path_point(self, point: np.ndarray) -> None:
        """把新 tip 点加入路径;与上一点过近则跳过(去重)。"""
        if np.linalg.norm(point - self._tip_path[-1]) > 1e-6:
            self._tip_path.append(np.asarray(point, dtype=float).reshape(3).copy())

    def _tip_path_length(self) -> float:
        if len(self._tip_path) < 2:
            return 0.0
        arr = np.asarray(self._tip_path)
        return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

    def _advance_tip_path(self, dist: float) -> bool:
        """让导丝沿"操作者意图方向"前进 dist。中心线仅作判断参考,不做自动牵引。

        决策分层(避免"一碰壁就停"):
          - aligned 层 (angle_to_centerline < align_angle_deg):
                混入 centerline_bias × center_tangent → centerline_assist。
          - slide-permission 层(更宽松,与 aligned 解耦):
                slide_allowed = (angle_to_centerline < slide_angle_max)
                              OR (projection_angle < yaw_slide_thresh)
                projection_angle = angle(project(raw_move_dir, wall_normal), raw_move_dir)
                即"投影后仍大致沿操作者意图"也认可,把壁面视作软栅栏。
          - 都不满足 → 真正 blocked,操作者必须手动转 J-tip。

        中心线**永远不**作为 wall-slide 方向源,滑动方向恒为 raw_move_dir 在壁面的投影。

        返回:
            True  — 推进成功,candidate 已加入 _tip_path。
            False — 被血管 SDF 拒绝,_tip_path 不变,调用方必须还原 qpos。
        """
        if dist <= 1e-9:
            return True
        tip = self._tip_path[-1]

        # ---- 1. raw_move_dir = 物理头方向 + J-tip steer(操作者意图)----
        physical_dir = self._estimate_tip_direction()
        raw_vec = np.asarray(physical_dir, dtype=float).copy()
        j_tip_w = self._j_tip_effective_weight()
        j_tip_dir = None
        if j_tip_w > 0.0:
            j_tip_dir = self._estimate_j_tip_direction(physical_dir)
            if j_tip_dir is not None:
                raw_vec = raw_vec + j_tip_w * j_tip_dir
        self._last_j_tip_weight = j_tip_w if j_tip_dir is not None else 0.0

        rn = float(np.linalg.norm(raw_vec))
        if not np.isfinite(rn) or rn < 1e-9:
            raw_move_dir = self.entry_dir.copy()
        else:
            raw_move_dir = raw_vec / rn

        # ---- 2. center_tangent + angle_to_centerline(只判断,不牵引)----
        center_tangent = self._nearest_centerline_tangent(tip)
        ctn = float(np.linalg.norm(center_tangent))
        if ctn > 1e-9:
            center_unit = center_tangent / ctn
            cos_a = float(np.clip(np.dot(raw_move_dir, center_unit), -1.0, 1.0))
            angle_deg = float(np.rad2deg(np.arccos(cos_a)))
        else:
            center_unit = raw_move_dir
            angle_deg = 0.0

        align_thresh = float(self.config.align_angle_deg)
        slide_angle_max = float(self.config.slide_angle_max)
        yaw_slide_thresh = float(self.config.yaw_slide_thresh)
        aligned = angle_deg < align_thresh

        # ---- 3. move_dir:aligned 与 yaw_tolerable_by_angle 都允许轻混 ----
        # (中心线只参考,不牵引;严重偏航 angle>=slide_angle_max 时不混)
        if angle_deg < slide_angle_max and ctn > 1e-9:
            bias = float(np.clip(self.config.centerline_bias, 0.0, 1.0))
            mv = (1.0 - bias) * raw_move_dir + bias * center_unit
            mn = float(np.linalg.norm(mv))
            move_dir = mv / mn if mn > 1e-9 else raw_move_dir
            status_intent = ("centerline_assist" if aligned
                             else "yaw_tolerable_blend")
        else:
            move_dir = raw_move_dir
            status_intent = "raw_forward"

        # 缓存 debug — 直推成功路径下保持这些值(slide 字段保持默认)
        self._last_angle_to_centerline = angle_deg
        self._last_raw_move_dir = raw_move_dir.copy()
        self._last_center_tangent = (
            center_unit.copy() if ctn > 1e-9 else None
        )
        self._last_move_dir = move_dir.copy()
        self._last_status_intent = status_intent
        self._last_slide_allowed = False
        self._last_slide_reason = "n/a"
        self._last_wall_normal = None
        self._last_candidate_slide_dir = None
        self._last_projection_angle_deg = float("nan")

        # ---- 4. 直推 ----
        candidate = tip + move_dir * dist
        if self._is_candidate_wire_safe(candidate):
            self._last_sdf_blocked = False
            self._last_wall_slide_attempted = False
            self._last_wall_slide_success = False
            self._last_status = status_intent
            self._append_tip_path_point(candidate)
            return True

        # ---- 5. 被壁面挡 → 计算 contact / wall_normal / candidate_slide_dir ----
        self._last_wall_slide_attempted = True
        contact = self._find_contact_point(tip, candidate)
        normal = (self._estimate_sdf_normal(contact)
                  if self._sdf_fn is not None else None)
        candidate_slide_dir: Optional[np.ndarray] = None
        projection_angle_deg = float("nan")
        if normal is not None:
            candidate_slide_dir = self._project_to_wall_tangent(raw_move_dir, normal)
            if candidate_slide_dir is not None:
                cp = float(np.clip(np.dot(candidate_slide_dir, raw_move_dir),
                                   -1.0, 1.0))
                projection_angle_deg = float(np.rad2deg(np.arccos(cp)))

        self._last_wall_normal = normal.copy() if normal is not None else None
        self._last_candidate_slide_dir = (
            candidate_slide_dir.copy()
            if candidate_slide_dir is not None else None
        )
        self._last_projection_angle_deg = projection_angle_deg

        # ---- 6. 双阈值决策 slide_reason / slide_allowed ----
        # 中心线偏角小 → 直接 aligned 或 yaw_tolerable(by-angle)。
        # 偏角大但投影方向仍接近 raw → yaw_tolerable(by-projection):墙是软栅栏。
        # 二者都不满足 → blocked。
        if angle_deg < align_thresh:
            slide_reason = "aligned"
        elif angle_deg < slide_angle_max:
            slide_reason = "yaw_tolerable"
        elif (candidate_slide_dir is not None
              and np.isfinite(projection_angle_deg)
              and projection_angle_deg < yaw_slide_thresh):
            slide_reason = "yaw_tolerable"
        else:
            slide_reason = "blocked"

        slide_allowed = slide_reason != "blocked"
        self._last_slide_reason = slide_reason
        self._last_slide_allowed = slide_allowed

        # ---- 7. 允许则用多候选 wall-slide 搜索(不是单方向单尝试)----
        # 即使 projection 失败(head-on,candidate_slide_dir is None),也允许进入
        # _try_wall_slide:它内部还有 last_slide_dir / 角度微扰候选可能救回来。
        if (slide_allowed
                and self.config.enable_wall_slide
                and self._sdf_fn is not None
                and normal is not None):
            ok = self._try_wall_slide(
                tip, move_dir, dist,
                slide_dir=candidate_slide_dir,  # 可以是 None,_try_wall_slide 自补
                contact=contact,
                normal=normal,
                raw_move_dir=raw_move_dir,
                allow_center_blend=(angle_deg < slide_angle_max),
            )
            if ok:
                self._last_sdf_blocked = False
                self._last_status = "wall_fallback"
                self._debug_step_block_reason(
                    stage="wall_slide_success", dist=dist,
                )
                return True
            # _try_wall_slide 内部 _last_wall_slide_reason 已写明失败原因
        else:
            self._last_wall_slide_reason = (
                "yaw_too_large" if not slide_allowed
                else ("no_normal" if normal is None else "disabled")
            )

        # ---- 8. blocked(本 substep)— 不是 sticky 状态,下一 substep 会重新尝试 ----
        self._last_sdf_blocked = True
        self._last_wall_slide_success = False
        self._last_status = "blocked"
        self._debug_step_block_reason(
            stage=("slide_candidates_unsafe" if slide_allowed
                   else ("yaw_too_large" if normal is not None
                         else "no_wall_normal")),
            dist=dist,
        )
        return False

    # ------------------------------------------------------------------
    # 调试:每次推进失败/滑动失败的完整状态打印(ZMD_GW_DEBUG=1 启用)
    # ------------------------------------------------------------------

    def _debug_step_block_reason(self, stage: str, dist: float) -> None:
        if not _DEBUG_STEP:
            return
        try:
            def _fmt(v):
                if v is None:
                    return "None"
                a = np.asarray(v, dtype=float).reshape(-1)
                return "(" + ",".join(f"{x:+.3f}" for x in a) + ")"
            print(
                f"[gw step] stage={stage} status={self._last_status} "
                f"intent={self._last_status_intent} "
                f"slide_reason={self._last_slide_reason} "
                f"slide_allowed={self._last_slide_allowed} "
                f"angle_to_cl={self._last_angle_to_centerline:+.1f}° "
                f"proj_angle={self._last_projection_angle_deg:+.1f}° "
                f"raw={_fmt(self._last_raw_move_dir)} "
                f"move={_fmt(self._last_move_dir)} "
                f"slide_cand={_fmt(self._last_candidate_slide_dir)} "
                f"wall_n={_fmt(self._last_wall_normal)} "
                f"sdf={self._last_min_sdf:+.4f} "
                f"step={dist:.4f} "
                f"twist={float(np.rad2deg(self._get_twist_angle())):+.1f}° "
                f"j_tip_w_eff={self._last_j_tip_weight:.3f} "
                f"j_tip_angle={float(self.config.j_tip_angle_deg):.1f}° "
                f"wall_slide_reason={self._last_wall_slide_reason}"
            )
        except Exception:
            pass

    def _retract_tip_path(self, dist: float) -> None:
        """沿导丝已走过的路径从尾部回退 dist(回到自己来时的路,而不是中心线)。"""
        remaining = float(dist)
        while remaining > 1e-9 and len(self._tip_path) > 1:
            seg = float(np.linalg.norm(self._tip_path[-1] - self._tip_path[-2]))
            if seg <= remaining + 1e-12:
                self._tip_path.pop()
                remaining -= seg
            else:
                t = (seg - remaining) / seg
                self._tip_path[-1] = (
                    self._tip_path[-2]
                    + t * (self._tip_path[-1] - self._tip_path[-2])
                )
                remaining = 0.0

    def _sample_tip_path_by_arclength(self, s: float) -> np.ndarray:
        """按弧长在导丝路径上采样点。s<=0 返回入口点;超出总长 clamp 到末端。"""
        path = self._tip_path
        if s <= 0.0 or len(path) < 2:
            return path[0].copy()
        acc = 0.0
        for j in range(1, len(path)):
            seg = float(np.linalg.norm(path[j] - path[j - 1]))
            if acc + seg >= s:
                t = (s - acc) / max(seg, 1e-9)
                return path[j - 1] + t * (path[j] - path[j - 1])
            acc += seg
        return path[-1].copy()

    def _rebuild_visible_wire_from_tip_path(self, path=None) -> np.ndarray:
        """从路径末端(tip)往后,按 segment_length 重建导丝可见点。

        path=None 时用 self._tip_path;也可传入临时 candidate path 做安全预检。
        s_i <= 0 的段还在入口外,沿 -entry_dir 直线放置;
        s_i > 0 的段在导丝路径上按弧长采样(不是中心线!)。
        """
        use_path = self._tip_path if path is None else path
        total = self._path_length(use_path)
        N = self.num_segments
        dx = self.segment_length
        pts = np.zeros((N, 3), dtype=float)
        for i in range(N):
            s_i = total - (N - 1 - i) * dx
            if s_i <= 0.0:
                pts[i] = self.entry_pos + s_i * self.entry_dir
            else:
                pts[i] = self._sample_path_by_arclength(use_path, s_i)
        return pts

    @staticmethod
    def _path_length(path) -> float:
        if path is None or len(path) < 2:
            return 0.0
        arr = np.asarray(path)
        return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

    @staticmethod
    def _sample_path_by_arclength(path, s: float) -> np.ndarray:
        """在给定 path 上按弧长 s 采样。s<=0 返回首点;超出总长 clamp 到末端。"""
        if s <= 0.0 or len(path) < 2:
            return np.asarray(path[0], dtype=float).copy()
        acc = 0.0
        for j in range(1, len(path)):
            a = np.asarray(path[j - 1], dtype=float)
            b = np.asarray(path[j], dtype=float)
            seg = float(np.linalg.norm(b - a))
            if acc + seg >= s:
                t = (s - acc) / max(seg, 1e-9)
                return a + t * (b - a)
            acc += seg
        return np.asarray(path[-1], dtype=float).copy()

    def _init_tip_path(self) -> None:
        """reset 时初始化路径为入口点;若有初始插入长度,沿弱引导推进建立初始路径。"""
        self._tip_path = [self.entry_pos.copy()]
        init = self.inserted_length
        if init > 1e-9:
            ds = max(0.25, self.segment_length * 0.25)
            remaining = init
            while remaining > 1e-9:
                step = min(ds, remaining)
                self._advance_tip_path(step)
                remaining -= step

    def _update_motion_debug(self) -> None:
        """更新弱引导调试信息(包括 SDF 碰撞与切向滑动状态)。"""
        tip = self._tip_path[-1]
        d = self._distance_to_centerline(tip)
        inserted = self.inserted_length
        travel = self._travel_length
        # 进度基于 tip 投影到中心线的弧长(单调),而不是 inserted_length。
        # 弱引导下导丝会偏离中心线,走过的实际路径 inserted 可能 >= travel,
        # 但 tip 在中心线投影上可能还没到终点 → 用 tip_s 算进度才正确。
        tip_s = self.tip_centerline_s()
        progress_by_tip = float(min(max(tip_s / travel, 0.0), 1.0)) if travel > 1e-9 else 0.0
        dist_end_by_tip = float(max(0.0, travel - tip_s))
        status = getattr(self, "_last_status", "raw_forward")
        # 新弱引导状态: "centerline_assist" / "raw_forward" / "wall_fallback" / "blocked"
        raw_dir = getattr(self, "_last_raw_move_dir", None)
        ctan = getattr(self, "_last_center_tangent", None)
        mdir = getattr(self, "_last_move_dir", None)
        wnorm = getattr(self, "_last_wall_normal", None)
        csdir = getattr(self, "_last_candidate_slide_dir", None)
        self.last_constraint_stats = {
            "mode": "weak_centerline",
            "status": status,
            "status_intent": str(getattr(self, "_last_status_intent", "raw_forward")),
            "angle_to_centerline": float(getattr(self, "_last_angle_to_centerline", 0.0)),
            "align_angle_deg": float(self.config.align_angle_deg),
            "slide_angle_max": float(self.config.slide_angle_max),
            "yaw_slide_thresh": float(self.config.yaw_slide_thresh),
            "slide_allowed": bool(getattr(self, "_last_slide_allowed", False)),
            "slide_reason": str(getattr(self, "_last_slide_reason", "n/a")),
            "projection_angle_deg": float(getattr(self, "_last_projection_angle_deg", float("nan"))),
            "raw_move_dir": (raw_dir.tolist() if raw_dir is not None else None),
            "center_tangent": (ctan.tolist() if ctan is not None else None),
            "wall_normal": (wnorm.tolist() if wnorm is not None else None),
            "candidate_slide_dir": (csdir.tolist() if csdir is not None else None),
            "move_dir": (mdir.tolist() if mdir is not None else None),
            "centerline_bias": float(np.clip(self.config.centerline_bias, 0.0, 1.0)),
            "tip_to_centerline_distance": float(d),
            "inserted_length": float(inserted),
            "travel_length": float(travel),
            # progress 基于 tip 投影弧长(UI 用这个)
            "progress": progress_by_tip,
            "tip_centerline_s": float(tip_s),
            "progress_by_tip": progress_by_tip,
            "distance_to_end_by_tip": dist_end_by_tip,
            # SDF 碰撞与切向滑动信息
            "min_sdf": float(getattr(self, "_last_min_sdf", float("nan"))),
            "sdf_enabled": self._sdf_fn is not None,
            "wall_slide_attempted": bool(getattr(self, "_last_wall_slide_attempted", False)),
            "wall_slide_success": bool(getattr(self, "_last_wall_slide_success", False)),
            "wall_slide_reason": str(getattr(self, "_last_wall_slide_reason", "none")),
            "sdf_normal": None,
            # J-tip 转向信息(确认 rotate 后真的影响方向)
            "twist_angle": float(self._get_twist_angle()),
            "j_tip_steering_enabled": bool(self.config.j_tip_steering_enabled),
            "j_tip_angle_deg": float(getattr(self.config, "j_tip_angle_deg", 0.0)),
            "j_tip_weight_effective": float(getattr(self, "_last_j_tip_weight", 0.0)),
            # 兼容旧字段(诊断面板/历史代码可能读)
            "max_radial_error": float(d),
            "num_projected_bodies": 0,
        }

    def _update_constraint_stats_kinematic(self) -> None:
        """[已弃用 — 第二步移除] 旧强贴合模式下的中心线偏离统计。现由 _update_motion_debug
        取代(报告 weak_centerline / tip_to_centerline_distance 等),本方法不再被调用。"""
        if self._tube_constraint is None or not self._seg_body_ids:
            return
        deployed = self._deployed_body_ids()
        if not deployed:
            self.last_constraint_stats = {
                "max_radial_error": 0.0,
                "num_projected_bodies": 0,
                "tip_centerline_s": 0.0,
            }
            return
        body_pos = np.asarray(self._data.xpos[deployed], dtype=float)
        idxs = self._tube_constraint._closest_idx(body_pos)
        radial = np.linalg.norm(body_pos - self._tube_constraint.points[idxs], axis=1)
        lumen = self._tube_constraint.lumen_radius[idxs]
        allowed = np.maximum(
            lumen - self._tube_constraint.guidewire_radius
            - self._tube_constraint.margin, 0.0
        )
        overshoot = np.maximum(radial - allowed, 0.0)
        self.last_constraint_stats = {
            "max_radial_error": float(overshoot.max()) if len(overshoot) else 0.0,
            "num_projected_bodies": int((overshoot > 0).sum()),
            "tip_centerline_s": float(self._tube_constraint.s[idxs[-1]]),
        }

    def _kinematic_mode_active(self) -> bool:
        return (
            self._centerline_points is not None
            and self.config.enable_centerline_constraint
        )

    def _deployed_body_ids(self) -> list[int]:
        """返回当前已送出 entry 的段 body ids(用 body xpos 在 entry_dir 方向上的投影判定)。"""
        if not self._seg_body_ids:
            return []
        ids = np.asarray(self._seg_body_ids, dtype=int)
        pos = np.asarray(self._data.xpos[ids], dtype=float)
        proj = (pos - self.entry_pos[None, :]) @ self.entry_dir
        mask = proj > 1e-6
        return [int(b) for b, m in zip(self._seg_body_ids, mask) if m]

    def _joint_qpos_addr(self, name: str) -> int:
        joint_id = self._mj.mj_name2id(
            self._model, self._mj.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            return -1
        return int(self._model.jnt_qposadr[joint_id])

    # ---------------------------------------------------------------- step

    def reset(self) -> None:
        self._mj.mj_resetData(self._model, self._data)
        if self._slide_qpos_idx >= 0:
            init = float(self.config.initial_inserted_length)
            init = float(np.clip(init, 0.0, self._max_inserted))
            self._data.qpos[self._slide_qpos_idx] = -self._max_length + init
        if self._twist_qpos_idx >= 0:
            self._data.qpos[self._twist_qpos_idx] = 0.0
        self._data.qvel[:] = 0.0
        self._mj.mj_forward(self._model, self._data)
        self.last_status = "idle"
        self.last_contact_stats = {}
        self.last_constraint_stats = {
            "max_radial_error": 0.0,
            "num_projected_bodies": 0,
            "tip_centerline_s": 0.0,
        }
        # 进度跟踪复位
        self._last_progress_s = 0.0
        self._last_motion_forward = True
        self._inset_camera_pos_smooth = None
        self._inset_focal_smooth = None
        self._inset_forward_smooth = None
        if self._kinematic_mode_active():
            # 弱引导:从入口沿自身方向(+轻微中心线引导)建立初始路径,不贴中心线
            self._init_tip_path()
            self._update_motion_debug()

    def forward(self, step_size: float = 1.0) -> bool:
        return self._apply_action(slide_delta=abs(float(step_size)))

    def backward(self, step_size: float = 1.0) -> bool:
        return self._apply_action(slide_delta=-abs(float(step_size)))

    def rotate_left(self, angle_rad: float = 0.1) -> bool:
        return self._apply_action(twist_delta=float(angle_rad))

    def rotate_right(self, angle_rad: float = 0.1) -> bool:
        return self._apply_action(twist_delta=-float(angle_rad))

    def step_simulation(self, n_steps: int = 3) -> bool:
        # 弱引导模式:不跑物理,也不贴中心线,只刷新调试状态
        if self._kinematic_mode_active():
            self._mj.mj_forward(self._model, self._data)
            self._update_motion_debug()
            self.last_status = "weak_centerline"
            return True
        steps = int(max(0, n_steps))
        for _ in range(steps):
            if hasattr(self._data, "xfrc_applied"):
                self._data.xfrc_applied[:] = 0.0
            # 中心线管腔约束:每步写 xfrc 拉回越界 body
            # 只对已经穿出 entry 的段生效(未送出的段在体外/护套里,不约束)
            if self._tube_constraint is not None and self._seg_body_ids:
                deployed = self._deployed_body_ids()
                stats = self._tube_constraint.step(
                    self._model, self._data, deployed
                )
                self.last_constraint_stats = stats
            self._mj.mj_step(self._model, self._data)
        if steps > 0:
            self.last_status = "free"
        # 收集 MuJoCo 内置接触统计(若有真实 hull collision)
        try:
            n_con = int(self._data.ncon)
            self.last_contact_stats = {"ncon": n_con}
        except Exception:
            self.last_contact_stats = {}
        return True

    def _apply_action(
        self, *, slide_delta: float = 0.0, twist_delta: float = 0.0
    ) -> bool:
        substeps = int(max(1, self.action_substeps))
        slide_step = float(slide_delta) / substeps
        twist_step = float(twist_delta) / substeps
        changed_any = False
        # 记录运动方向(进度单调跟踪用):>=0 视为前进
        if slide_delta < 0:
            self._last_motion_forward = False
        elif slide_delta > 0:
            self._last_motion_forward = True

        for _ in range(substeps):
            changed = False
            slide_actual = 0.0
            prev_slide_qpos = None  # 保存原始 qpos，SDF 拒绝时用于还原

            if slide_step != 0.0 and self._slide_qpos_idx >= 0:
                current = float(self._data.qpos[self._slide_qpos_idx])
                updated = float(
                    np.clip(
                        current + slide_step,
                        -self._max_length,
                        -self._max_length + self._max_inserted,
                    )
                )
                if abs(updated - current) > 1e-12:
                    prev_slide_qpos = current   # 保存，以便 SDF 拒绝时回滚
                    self._data.qpos[self._slide_qpos_idx] = updated
                    slide_actual = updated - current
                    changed = True

            if twist_step != 0.0 and self._twist_qpos_idx >= 0:
                self._data.qpos[self._twist_qpos_idx] += twist_step
                changed = True

            if not changed:
                continue

            self._data.qvel[:] = 0.0
            if self._kinematic_mode_active():
                # 弱引导模式:导丝按自身方向推进/沿自身路径回退,中心线只做方向参考。
                # 不再 pre-shape 把链贴到 centerline 上,tip 不会被 teleport 到中心线点。
                if slide_actual > 1e-12:
                    allowed = self._advance_tip_path(slide_actual)
                    if not allowed:
                        # SDF 拦截:还原 qpos,保持导丝不动
                        if prev_slide_qpos is not None and self._slide_qpos_idx >= 0:
                            self._data.qpos[self._slide_qpos_idx] = prev_slide_qpos
                        # _tip_path 未被修改(由 _advance_tip_path 保证),无需额外处理
                elif slide_actual < -1e-12:
                    # backward 不检查 SDF(沿已走过的安全路径回退)
                    self._retract_tip_path(-slide_actual)
                    self._last_sdf_blocked = False
                else:
                    self._last_sdf_blocked = False
                self._mj.mj_forward(self._model, self._data)
                self._update_motion_debug()
            else:
                # 物理模式:无 centerline 时回退到力学 + xfrc 约束
                self._mj.mj_forward(self._model, self._data)
                self.step_simulation(self.mj_substeps_per_action)
            changed_any = True

        if changed_any:
            self.last_status = "free"
        return changed_any

    # --------------------------------------------------------------- query

    @property
    def inserted_length(self) -> float:
        if self._slide_qpos_idx < 0:
            return 0.0
        return float(
            np.clip(
                self._data.qpos[self._slide_qpos_idx] + self._max_length,
                0.0,
                self._max_inserted,
            )
        )

    @property
    def rotation_angle(self) -> float:
        if self._twist_qpos_idx < 0:
            return 0.0
        return float(self._data.qpos[self._twist_qpos_idx])

    @property
    def max_length(self) -> float:
        """导丝自身物理长度(段数 × 段长)。"""
        return self._max_length

    @property
    def travel_length(self) -> float:
        """可推进的总路程 = 中心线起点→终点弧长(无中心线时退化为导丝长度)。"""
        return self._travel_length

    @property
    def loaded_hull_count(self) -> int:
        return int(self._loaded_hull_count)

    def get_tip_position(self) -> np.ndarray:
        if self.inserted_length <= 1e-9:
            return self.entry_pos.copy()
        # 弱引导模式:tip = 导丝实际路径末端(不是中心线点)
        if self._kinematic_mode_active():
            return np.asarray(self._tip_path[-1], dtype=float).copy()
        if not self._site_ids:
            return self.entry_pos.copy()
        return np.asarray(self._data.site_xpos[self._site_ids[-1]], dtype=float).copy()

    def get_tip_direction(self) -> np.ndarray:
        # 弱引导模式:用导丝路径估计头部方向
        if self._kinematic_mode_active():
            return self._estimate_tip_direction()
        if len(self._site_ids) < 2:
            return self.entry_dir.copy()
        prev_point = np.asarray(self._data.site_xpos[self._site_ids[-2]], dtype=float)
        tip_point = np.asarray(self._data.site_xpos[self._site_ids[-1]], dtype=float)
        tangent = tip_point - prev_point
        norm = float(np.linalg.norm(tangent))
        return tangent / norm if norm > 1e-9 else self.entry_dir.copy()

    # ---------------------------------------------------------- inset camera

    def compute_inset_camera_pose(self, override_tip=None, override_forward=None):
        """导丝头部"第一人称跟随"相机位姿。

        语义:相机绑在 tip 后方 back_distance,始终朝导丝当前 heading 方向看。
            camera_position = tip - forward * back_distance
            camera_focal    = tip + forward * lookahead_distance

        override_tip / override_forward (可选):wrapper 传入"用户视觉上看到的 tip
        + tangent"(例如 J-tip overlay 的末端),让相机跟踪视觉 tip 而不是 backend
        物理 tip。twist 旋转改变视觉 tip 位置 → 相机跟着扫,符合"始终跟踪视角"。

        默认 forward = 导丝自身 heading(_estimate_tip_direction:由 tip_path 末端
        一小段差分得到)。** 故意不用中心线切向 ** — 那会变成"血管视角"。

        低通滤波 forward / cam_pos / focal 抑制 substep 抖动。
        未插入时返回 None,inset 端会显示"未插入导丝"占位。
        """
        if self.inserted_length <= 1e-9:
            return None
        if override_tip is not None:
            tip = np.asarray(override_tip, dtype=float).reshape(3)
        else:
            tip = self.get_tip_position()

        # forward:优先调用方传入的视觉 tangent(随 J-tip 旋转变),否则用物理 heading
        if override_forward is not None:
            fwd = np.asarray(override_forward, dtype=float).reshape(3)
        else:
            fwd = self._estimate_tip_direction()
        ln = float(np.linalg.norm(fwd))
        if not np.isfinite(ln) or ln < 1e-9:
            fwd = self.entry_dir.copy()
            ln = float(np.linalg.norm(fwd))
        fwd = fwd / ln if ln > 1e-9 else self.entry_dir.copy()

        # 翻转保护(分叉处切向符号跳变 → 与上一帧对齐)
        if (self._inset_forward_smooth is not None
                and float(np.dot(self._inset_forward_smooth, fwd)) < 0.0):
            fwd = -fwd

        # heading 单独重平滑(0.85,~6 tick 时间常数),抑制 substep 噪声让视向稳定。
        # 相机位置和焦点用更轻的 inset_camera_smooth(0.2)保持跟手。
        heading_a = float(np.clip(self.config.inset_heading_smooth, 0.0, 0.99))
        if self._inset_forward_smooth is None:
            fwd_s = fwd
        else:
            fwd_s = heading_a * self._inset_forward_smooth + (1.0 - heading_a) * fwd
            n = float(np.linalg.norm(fwd_s))
            fwd_s = fwd_s / n if n > 1e-9 else fwd
        self._inset_forward_smooth = fwd_s
        a = float(np.clip(self.config.inset_camera_smooth, 0.0, 0.99))

        # up 向量:与 fwd_s 正交,优先世界 Z(退化时用世界 Y)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(world_up, fwd_s))) > 0.95:
            world_up = np.array([0.0, 1.0, 0.0])
        side = np.cross(fwd_s, world_up)
        sn = float(np.linalg.norm(side))
        side = side / sn if sn > 1e-9 else np.array([1.0, 0.0, 0.0])
        up = np.cross(side, fwd_s)
        un = float(np.linalg.norm(up))
        up = up / un if un > 1e-9 else np.array([0.0, 0.0, 1.0])

        back = float(self.config.inset_camera_back_offset)
        up_off = float(self.config.inset_camera_up_offset)

        # 动态跟随观察:cam 在 tip 后方 back + 上方 up_off;focal **始终锚定 tip**。
        # focal=tip 而不是 tip+fwd*fdist 的关键:fwd 即便仍有微抖,焦点也不动,
        # tip 永远在画面中央,看上去就是"动态保持看着导丝头",不晃。
        target_cam = tip - fwd_s * back + up * up_off
        target_focal = tip

        # 低通 cam_pos / focal
        if self._inset_camera_pos_smooth is None:
            cam_pos = target_cam
            focal = target_focal
        else:
            cam_pos = a * self._inset_camera_pos_smooth + (1.0 - a) * target_cam
            focal = a * self._inset_focal_smooth + (1.0 - a) * target_focal
        self._inset_camera_pos_smooth = np.asarray(cam_pos, dtype=float).copy()
        self._inset_focal_smooth = np.asarray(focal, dtype=float).copy()

        return {
            "position": np.asarray(cam_pos, dtype=float),
            "focal_point": np.asarray(focal, dtype=float),
            "up": np.asarray(up, dtype=float),
        }

    def get_guidewire_points(self) -> np.ndarray:
        if self.inserted_length <= 1e-9:
            return np.empty((0, 3), dtype=float)

        # 弱引导模式:导丝可见点从自身路径(_tip_path)回填,不再来自中心线/MuJoCo site
        if self._kinematic_mode_active():
            all_points = self._rebuild_visible_wire_from_tip_path()
        else:
            if not self._site_ids:
                return np.empty((0, 3), dtype=float)
            all_points = np.asarray(
                [self._data.site_xpos[site_id] for site_id in self._site_ids],
                dtype=float,
            )
        # entry_pos 到 entry_dir 投影:>0 表示已经从入口送出
        projections = (all_points - self.entry_pos[None, :]) @ self.entry_dir
        deployed = all_points[projections > 1e-6]
        if len(deployed) == 0:
            return np.vstack([self.entry_pos[None, :], self.get_tip_position()[None, :]])
        return np.vstack([self.entry_pos[None, :], deployed])

    def close(self) -> None:
        self._model = None
        self._data = None
