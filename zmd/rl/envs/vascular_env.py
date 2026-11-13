"""血管导丝导航环境 — gymnasium 风格,给 sb3 PPO 用。

每个 env 实例持有**独立的** MujocoGuidewire(独立 MuJoCo model/data),与 UI
里的导丝完全隔离,所以训练线程不会和渲染竞态。

动作空间(Discrete 4)— 移除了 NOOP,强制每步必须做点什么,杜绝末段"原地摆烂":
    0 = ADVANCE          前进 1 步(MujocoGuidewire.advance_step)
    1 = RETRACT          后退 1 步
    2 = ROTATE_CW        顺时针转 J-tip
    3 = ROTATE_CCW       逆时针转 J-tip

观测(Box 20-d float32):
    tip_pos_norm(3) + tip_dir(3) + center_tangent(3) +
    angle_to_cl/180(1) + min_sdf_clip(1) + inserted_norm(1) +
    progress(1) + dist_to_end_norm(1) + sin(twist)/cos(twist)(2) +
    status_intent_one_hot(4)

奖励:由 reward_params(dict) 配置,与 RLPanel 的默认配方一致。
终止:progress > success_progress(默认 0.92)。
截断:step_count >= max_steps(env 构造参数)。
"""
from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from zmd.rl.speed_schedule import resolve_speed_config
from zmd.simulation.guidewire.mujoco_config import MujocoGuidewireConfig
from zmd.simulation.guidewire.mujoco_guidewire import MujocoGuidewire


ACTION_ADVANCE   = 0
ACTION_RETRACT   = 1
ACTION_ROTATE_CW = 2
ACTION_ROTATE_CCW = 3
# 注:NOOP 已废弃(action_space=Discrete(4));保留常量给旧代码引用不会崩
ACTION_NOOP      = 4
ACTION_NAMES = {
    ACTION_ADVANCE: "ADVANCE",
    ACTION_RETRACT: "RETRACT",
    ACTION_ROTATE_CW: "ROTATE_CW",
    ACTION_ROTATE_CCW: "ROTATE_CCW",
    ACTION_NOOP: "NOOP",
}

STATUS_INTENT_INDEX = {
    "centerline_assist":    0,
    "raw_forward":          1,
    "yaw_tolerable_blend":  2,
    "wall_fallback":        3,
}

# 默认奖励参数(必须与 RLPanel.DEFAULT_REWARD_PARAMS 同步)
DEFAULT_REWARD_PARAMS = {
    "progress_weight":   200.0,
    "backtrack_weight":  100.0,
    "step_cost":         0.02,
    "blocked_penalty":   0.5,
    "retract_cost":      0.1,
    "rotate_cost":       0.05,
    "yaw_soft_thresh":   30.0,
    "yaw_soft_weight":   0.01,
    "safety_margin_mm":  0.5,
    "safety_weight":     0.5,
    "success_bonus":     300.0,
    "timeout_penalty":   50.0,
    "success_progress":  0.92,
}


# Optional guidewire physics overrides for controlled curriculum experiments.
# Empty by default so RL training uses the same MuJoCoGuidewireConfig as UI.
DEFAULT_GUIDEWIRE_PARAMS = {}


def compute_progress_reward(
    progress: float,
    last_progress: float,
    best_progress: float,
    reward_params: dict,
) -> tuple[dict[str, float], float]:
    """Return anti-cycling progress terms and the updated best progress.

    Only progress beyond the episode's previous frontier earns a positive
    reward. Moving backwards is penalized, and moving forward over an already
    visited segment earns no second progress reward. This makes repeated
    advance/retract cycles unprofitable while preserving a dense signal when
    the tip reaches genuinely new parts of the route.
    """
    progress = float(progress)
    last_progress = float(last_progress)
    best_progress = float(best_progress)
    new_progress = max(0.0, progress - best_progress)
    backtrack = max(0.0, last_progress - progress)
    components = {
        "progress": float(reward_params["progress_weight"]) * new_progress,
        "backtrack": -float(reward_params.get("backtrack_weight", 0.0)) * backtrack,
    }
    return components, max(best_progress, progress)


def build_observation(gw, bbox_center, bbox_extent, max_length, travel_length) -> np.ndarray:
    """从任意 MujocoGuidewire + vessel 尺度信息构造 20-d 观测。

    训练时由 VascularNavEnv 调用,推理回放时由 PPOInferenceController 在 UI wire 上调用 ——
    保证两边的 obs 完全一致(model.predict 的输入分布 = 训练时见过的分布)。
    """
    backend = gw.backend
    stats = backend.last_constraint_stats

    tip = backend.get_tip_position()
    tip_norm = (tip - bbox_center) / bbox_extent

    fwd = backend._estimate_tip_direction()
    ct = backend._nearest_centerline_tangent(tip)

    angle_n = float(stats.get("angle_to_centerline", 0.0)) / 180.0
    min_sdf = float(stats.get("min_sdf", 0.0))
    if not math.isfinite(min_sdf):
        min_sdf = 0.0
    min_sdf = max(-2.0, min(0.0, min_sdf))
    inserted_n = float(backend.inserted_length) / max(float(max_length), 1e-3)
    progress = float(stats.get("progress", 0.0))
    dend_n = float(stats.get("distance_to_end_by_tip", 0.0)) / max(float(travel_length), 1e-3)

    twist = float(backend.rotation_angle)

    intent = str(stats.get("status_intent", "raw_forward"))
    intent_idx = STATUS_INTENT_INDEX.get(intent, 1)
    intent_oh = np.zeros(4, dtype=np.float32)
    intent_oh[intent_idx] = 1.0

    obs = np.concatenate([
        tip_norm.astype(np.float32),
        fwd.astype(np.float32),
        ct.astype(np.float32),
        np.array([angle_n, min_sdf, inserted_n, progress, dend_n],
                 dtype=np.float32),
        np.array([math.sin(twist), math.cos(twist)], dtype=np.float32),
        intent_oh,
    ])
    return obs.astype(np.float32)


class VascularNavEnv(gym.Env):
    """导丝导航环境,标准 gymnasium 接口。

    支持单路线(centerline=1 条)或多路线泛化(centerlines=N 条):
    - 单路线:常规训练,reset 总用同一条 centerline + 同一个 wire 实例
    - 多路线:reset 随机抽一条 centerline,切换对应预建好的 wire 实例
      (每条路线 entry/exit 不同,wire 的 MuJoCo body chain 锚点不同,
      必须独立 wire 实例;切换时只换 active 引用,不重建 MJCF)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        centerline=None,
        centerlines=None,
        sdf_fn=None,
        reward_params: Optional[dict] = None,
        guidewire_params: Optional[dict] = None,
        speed_config=None,
        max_steps: int = 1200,
        success_progress: float = 0.92,   # 训练和推理默认使用同一个到达终点阈值
                                           # 触发 success bonus,给 PPO 一个能学到的目标信号
    ):
        super().__init__()
        # 统一存成 list,单路线就是 list 长度 1
        if centerlines is not None:
            self._routes = list(centerlines)
        elif centerline is not None:
            self._routes = [centerline]
        else:
            raise ValueError("VascularNavEnv requires centerline or centerlines")
        self.sdf_fn = sdf_fn
        self.reward_params = {**DEFAULT_REWARD_PARAMS, **(reward_params or {})}
        self.guidewire_params = {
            **DEFAULT_GUIDEWIRE_PARAMS,
            **(guidewire_params or {}),
        }
        self.speed_config = resolve_speed_config(speed_config)
        self.max_steps = int(max_steps)
        self.success_progress = float(
            self.reward_params.get("success_progress", success_progress)
        )

        # 每条路线预建独立 wire(MuJoCo body chain 因 entry/dir 不同要分开构造)
        self._wires = []
        wire_config = MujocoGuidewireConfig().copy(**self.guidewire_params)
        for cl in self._routes:
            gw = MujocoGuidewire(cl, on_state_change=None, config=wire_config)
            if sdf_fn is not None and hasattr(gw, "set_collision_sdf"):
                gw.set_collision_sdf(sdf_fn)
            self._wires.append(gw)

        # 每条路线的 bbox / 尺度缓存(obs 归一化用)
        self._route_meta = []
        for cl, gw in zip(self._routes, self._wires):
            pts = np.asarray(cl.points, dtype=float)
            self._route_meta.append({
                "bbox_center": (pts.min(0) + pts.max(0)) * 0.5,
                "bbox_extent": np.maximum(pts.max(0) - pts.min(0), 1e-3),
                "travel_length": float(getattr(gw.backend, "_travel_length", 1.0)),
                "max_length": float(getattr(gw.backend, "_max_length", 1.0)),
            })

        self._route_idx = 0
        self._refresh_active_route()
        self._safe_threshold = float(self.gw.radius) + float(
            self.gw.config.collision_margin
        )

        self.action_space = spaces.Discrete(4)   # ADVANCE / RETRACT / ROTATE_CW / ROTATE_CCW
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32,
        )

        self._step_count = 0
        self._last_progress = 0.0
        self._best_progress = 0.0
        self._distal_speed_active = False
        # 训练时可视化用:每 step 调 observer(polyline, camera_pose, info)
        # observer 由 trainer 设置,内部节流后 emit Qt signal 到 UI 主线程。
        self.step_observer = None

    @property
    def gw(self):
        """当前 active wire(reset 时切换)。"""
        return self._wires[self._route_idx]

    @property
    def centerline(self):
        return self._routes[self._route_idx]

    @property
    def n_routes(self) -> int:
        return len(self._routes)

    def _refresh_active_route(self):
        """切换到 _route_idx 指向的路线,刷新 bbox/尺度缓存。"""
        m = self._route_meta[self._route_idx]
        self._bbox_center = m["bbox_center"]
        self._bbox_extent = m["bbox_extent"]
        self._travel_length = m["travel_length"]
        self._max_length = m["max_length"]

    def _notify_observer(self, info: dict):
        if self.step_observer is None:
            return
        try:
            polyline = self.gw.backend.get_guidewire_points()
            cam = self.gw.get_inset_camera_pose()
            self.step_observer(polyline, cam, info)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # gym 接口
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 多路线:每个 episode 随机抽一条作为本集训练路线
        if len(self._wires) > 1:
            self._route_idx = int(self.np_random.integers(0, len(self._wires)))
            self._refresh_active_route()
        self.gw.reset()
        self._step_count = 0
        self._last_progress = 0.0
        self._best_progress = 0.0
        self._distal_speed_active = False
        # 每集统计 — 给 trainer 的 callback 在 episode 结束时读
        self._ep_blocked_count = 0
        self._ep_blocked_advance_count = 0
        self._ep_worst_sdf = float("-inf")
        self._ep_action_counts = {index: 0 for index in range(4)}
        self._ep_proximal_advance_count = 0
        self._ep_distal_advance_count = 0
        self._ep_requested_advance_mm = 0.0
        self._ep_reward_components = {
            key: 0.0 for key in (
                "progress", "backtrack", "blocked", "retract", "rotate",
                "yaw", "safety", "step", "terminal",
            )
        }
        obs = self._observation()
        self._notify_observer({"reset": True, "route_idx": self._route_idx})
        return obs, {}

    def step(self, action):
        action = int(action)
        speed_phase = self.speed_config.phase(
            self._best_progress,
            self._distal_speed_active,
        )
        if speed_phase == "distal":
            self._distal_speed_active = True
        requested_push_mm = 0.0
        action_applied = False
        if action == ACTION_ADVANCE:
            requested_push_mm = self.speed_config.push_step_mm(
                self._best_progress,
                self._distal_speed_active,
            )
            action_applied = bool(self.gw.advance_by(requested_push_mm))
        elif action == ACTION_RETRACT:
            # The A-series isolates the forward-speed intervention.  Retraction
            # keeps the guidewire's original 0.50 mm step.
            action_applied = bool(self.gw.retract_step())
        elif action == ACTION_ROTATE_CW:
            action_applied = bool(self.gw.rotate_clockwise())
        elif action == ACTION_ROTATE_CCW:
            action_applied = bool(self.gw.rotate_counterclockwise())
        # action_space = Discrete(4) 后,model 不会输出 4(NOOP);其它非法值忽略

        return self._finish_step(
            action=action,
            action_applied=action_applied,
            requested_push_mm=requested_push_mm,
            speed_phase=speed_phase,
        )

    def _finish_step(
        self,
        action: int,
        action_applied: bool,
        requested_push_mm: float = 0.0,
        speed_phase: str = "proximal",
    ):
        stats = self.gw.backend.last_constraint_stats
        progress = float(stats.get("progress", 0.0))
        progress_components, self._best_progress = compute_progress_reward(
            progress=progress,
            last_progress=self._last_progress,
            best_progress=self._best_progress,
            reward_params=self.reward_params,
        )
        self._last_progress = progress

        # 累积每集统计(blocked 计数 / 最差 SDF 余量)
        if stats.get("status") == "blocked":
            self._ep_blocked_count += 1
        blocked_this_step = bool(action == ACTION_ADVANCE and not action_applied)
        if blocked_this_step:
            self._ep_blocked_advance_count += 1
        if action in self._ep_action_counts:
            self._ep_action_counts[action] += 1
        if action == ACTION_ADVANCE:
            self._ep_requested_advance_mm += float(requested_push_mm)
            if speed_phase == "distal":
                self._ep_distal_advance_count += 1
            else:
                self._ep_proximal_advance_count += 1
        cur_sdf = float(stats.get("min_sdf", float("nan")))
        if math.isfinite(cur_sdf):
            self._ep_worst_sdf = max(self._ep_worst_sdf, cur_sdf)

        # ---- reward ----
        rp = self.reward_params
        reward_components = {
            **progress_components,
            "blocked": 0.0,
            "retract": 0.0,
            "rotate": 0.0,
            "yaw": 0.0,
            "safety": 0.0,
            "step": -float(rp["step_cost"]),
            "terminal": 0.0,
        }
        if stats.get("status") == "blocked":
            reward_components["blocked"] = -float(rp["blocked_penalty"])
        if action == ACTION_RETRACT:
            reward_components["retract"] = -float(rp.get("retract_cost", 0.0))
        if action in (ACTION_ROTATE_CW, ACTION_ROTATE_CCW):
            reward_components["rotate"] = -float(rp["rotate_cost"])
        angle = float(stats.get("angle_to_centerline", 0.0))
        if angle > rp["yaw_soft_thresh"]:
            reward_components["yaw"] = (
                -float(rp["yaw_soft_weight"]) * (angle - float(rp["yaw_soft_thresh"]))
            )
        # SDF 安全余量
        min_sdf = float(stats.get("min_sdf", float("nan")))
        if math.isfinite(min_sdf):
            safety = -self._safe_threshold - min_sdf
            if safety < rp["safety_margin_mm"]:
                reward_components["safety"] = (
                    -float(rp["safety_weight"])
                    * (float(rp["safety_margin_mm"]) - max(0.0, safety))
                )
        else:
            safety = float("nan")

        # ---- terminate / truncate ----
        self._step_count += 1
        terminated = bool(progress >= self.success_progress)
        truncated = bool(self._step_count >= self.max_steps)
        if terminated:
            reward_components["terminal"] = float(rp["success_bonus"])
        elif truncated:
            reward_components["terminal"] = -float(rp["timeout_penalty"])

        r = float(sum(reward_components.values()))
        for key, value in reward_components.items():
            self._ep_reward_components[key] += float(value)

        info = {
            "progress": progress,
            "best_progress": float(self._best_progress),
            "status": stats.get("status"),
            "success": terminated,
            "action_index": int(action),
            "action_name": ACTION_NAMES.get(int(action), f"UNKNOWN_{action}"),
            "action_applied": bool(action_applied),
            "blocked_this_step": blocked_this_step,
            "speed_profile": self.speed_config.profile,
            "speed_phase": str(speed_phase),
            "requested_push_mm": float(requested_push_mm),
            "tip_centerline_s": float(stats.get("tip_centerline_s", 0.0)),
            "distance_to_end_by_tip": float(stats.get("distance_to_end_by_tip", 0.0)),
            "angle_to_centerline": float(stats.get("angle_to_centerline", 0.0)),
            "tip_to_centerline_distance": float(
                stats.get("tip_to_centerline_distance", 0.0)
            ),
            "min_sdf": min_sdf,
            "safety_margin_mm": safety,
            "wall_slide_attempted": bool(stats.get("wall_slide_attempted", False)),
            "wall_slide_success": bool(stats.get("wall_slide_success", False)),
            "wall_slide_reason": str(stats.get("wall_slide_reason", "none")),
            "twist_angle": float(stats.get("twist_angle", 0.0)),
            "ep_length": int(self._step_count),
            "ep_blocked": int(self._ep_blocked_count),
            "ep_blocked_advance": int(self._ep_blocked_advance_count),
            "ep_advance": int(self._ep_action_counts[ACTION_ADVANCE]),
            "ep_retract": int(self._ep_action_counts[ACTION_RETRACT]),
            "ep_rotate_cw": int(self._ep_action_counts[ACTION_ROTATE_CW]),
            "ep_rotate_ccw": int(self._ep_action_counts[ACTION_ROTATE_CCW]),
            "ep_proximal_advance": int(self._ep_proximal_advance_count),
            "ep_distal_advance": int(self._ep_distal_advance_count),
            "ep_requested_advance_mm": float(self._ep_requested_advance_mm),
            "ep_worst_sdf": float(self._ep_worst_sdf if math.isfinite(self._ep_worst_sdf) else 0.0),
            "reward_components": dict(reward_components),
            "ep_reward_components": dict(self._ep_reward_components),
            "route_idx": int(self._route_idx),
        }
        obs = self._observation()
        self._notify_observer(info)
        return obs, float(r), terminated, truncated, info

    # ------------------------------------------------------------------
    # 观测构造
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        return build_observation(
            self.gw, self._bbox_center, self._bbox_extent,
            self._max_length, self._travel_length,
        )

    def close(self):
        for gw in self._wires:
            try:
                gw.close()
            except Exception:
                pass


class VascularNavContinuousEnv(VascularNavEnv):
    """Box-action variant for SAC/TD3/DDPG style algorithms.

    The reward, observation, and executed primitives match VascularNavEnv.
    SAC still emits a continuous Box action, but the environment maps it to
    exactly one PPO-style primitive per step:
    - dominant slide axis: negative=retract, positive=advance
    - dominant rotate axis: negative=CCW, positive=CW
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    def step(self, action):
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        slide = float(np.clip(arr[0] if len(arr) >= 1 else 0.0, -1.0, 1.0))
        rotate = float(np.clip(arr[1] if len(arr) >= 2 else 0.0, -1.0, 1.0))

        if abs(slide) >= abs(rotate):
            if slide >= 0.0:
                primitive = ACTION_ADVANCE
            else:
                primitive = ACTION_RETRACT
        else:
            if rotate >= 0.0:
                primitive = ACTION_ROTATE_CW
            else:
                primitive = ACTION_ROTATE_CCW

        return super().step(primitive)
