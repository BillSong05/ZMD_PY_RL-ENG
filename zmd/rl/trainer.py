"""PPO 训练器 — 把 stable-baselines3 PPO 接入 Qt 信号槽。

并发模型:
- 单独 QThread 跑 model.learn() (sb3 训练循环阻塞,必须独立线程)
- 训练 env 持有**独立** MujocoGuidewire,不和 UI 渲染竞态
- 通过 sb3 callback 把 episode 统计 emit 回主线程 UI
- stop_requested 标志在 callback 里检查,优雅中断

ckpt 存放:projects/cache/rl/ppo_<algo>_<timestamp>.zip
"""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from stable_baselines3 import PPO, SAC, TD3, DDPG, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

from zmd.rl.envs.vascular_env import (
    DEFAULT_GUIDEWIRE_PARAMS,
    VascularNavContinuousEnv,
    VascularNavEnv,
)
from zmd.rl.speed_schedule import resolve_speed_config
from zmd.simulation.guidewire.mujoco_config import MujocoGuidewireConfig


REWARD_COMPONENT_KEYS = (
    "progress", "backtrack", "blocked", "retract", "rotate",
    "yaw", "safety", "step", "terminal",
)


class _PPOCallback(BaseCallback):
    """sb3 回调:发送 episode 统计 + 检查停止标志 + max_episodes 控流。"""

    def __init__(self, trainer: "PPOTrainer"):
        super().__init__()
        self.trainer = trainer
        self.episode_count = 0
        self._recent_rewards: list[float] = []
        self._recent_successes: list[float] = []
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        # locals['infos'] 是 sb3 标准接口
        for info in self.locals.get("infos", []):
            ep = info.get("episode") if info else None
            if ep is None:
                continue
            self.episode_count += 1
            ep_r = float(ep.get("r", 0.0))
            success = bool(info.get("success", False))
            self._recent_rewards.append(ep_r)
            self._recent_successes.append(1.0 if success else 0.0)
            self._recent_rewards = self._recent_rewards[-50:]
            self._recent_successes = self._recent_successes[-50:]
            payload = {
                "index":           int(self.episode_count),
                "timestep":        int(self.num_timesteps),
                "reward":          ep_r,
                "avg_reward_50":   float(np.mean(self._recent_rewards)),
                "success":         success,
                "success_rate_50": float(np.mean(self._recent_successes)),
                "length":          int(info.get("ep_length", int(ep.get("l", 0)))),
                "blocked":         int(info.get("ep_blocked", 0)),
                "worst_sdf":       float(info.get("ep_worst_sdf", 0.0)),
                "final_progress":   float(info.get("progress", 0.0)),
                "max_progress":     float(info.get("best_progress", 0.0)),
                "route_idx":       int(info.get("route_idx", 0)),
                "blocked_advance": int(info.get("ep_blocked_advance", 0)),
                "advance":         int(info.get("ep_advance", 0)),
                "retract":         int(info.get("ep_retract", 0)),
                "rotate_cw":       int(info.get("ep_rotate_cw", 0)),
                "rotate_ccw":      int(info.get("ep_rotate_ccw", 0)),
                "proximal_advance": int(info.get("ep_proximal_advance", 0)),
                "distal_advance":  int(info.get("ep_distal_advance", 0)),
                "requested_advance_mm": float(
                    info.get("ep_requested_advance_mm", 0.0)
                ),
            }
            reward_parts = info.get("ep_reward_components", {})
            for key in REWARD_COMPONENT_KEYS:
                payload[f"reward_{key}"] = float(reward_parts.get(key, 0.0))
            self.trainer._write_episode_log(payload)
            self.trainer.episode_ended.emit(payload)

        # 周期性评估(论文式曲线数据源)
        if (self.trainer.eval_freq > 0
                and self.num_timesteps - self._last_eval_step >= self.trainer.eval_freq):
            self._last_eval_step = self.num_timesteps
            try:
                self.trainer._run_and_emit_evaluation(self.model, int(self.num_timesteps))
            except Exception:
                import traceback; traceback.print_exc()

        if self.trainer.stop_requested:
            return False  # 中止 learn()
        if self.episode_count >= self.trainer.max_episodes:
            return False
        return True


class PPOTrainer(QObject):
    """sb3 PPO 训练器。move_to_thread 后 run() 跑训练循环。

    Signals:
        episode_ended(ep_idx, ep_reward, avg_reward_50, success_rate_50)
        status_changed(text)
        finished(ckpt_path)   ckpt_path 为空字符串表示失败/取消
        step_observed(polyline: np.ndarray, camera_pose: object)
            每 ~40ms(节流) emit 当前 wire 状态供 UI 实时渲染。
            camera_pose 是 dict 或 None(np 数组已 .copy() 跨线程安全)。
    """

    # episode_ended:训练态文字用(每个训练 episode);curves 不用它(带探索噪声)
    episode_ended = pyqtSignal(dict)
    # eval_done:论文式评估曲线数据点。每 eval_freq 训练步用 deterministic policy
    # 跑 n_eval 个回合统计 → {timestep, success_rate, mean_reward, mean_length, mean_blocked}
    eval_done = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal(str)
    step_observed = pyqtSignal(object, object)   # polyline copy, cam_pose dict|None
    algo_name = "ppo"
    display_name = "PPO"
    env_class = VascularNavEnv

    def __init__(
        self,
        centerline=None,
        sdf_fn=None,
        reward_params: dict = None,
        max_episodes: int = 500,
        max_steps_per_episode: int = 1200,
        ckpt_dir: str = "projects/cache/rl",
        centerlines=None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 1,
        seed: int = 0,
        speed_config=None,
        parent=None,
    ):
        super().__init__(parent)
        # 支持单路线 centerline 或多路线 centerlines(泛化训练)
        self.centerline = centerline
        self.centerlines = list(centerlines) if centerlines else None
        self.sdf_fn = sdf_fn
        self.sdf_enabled = sdf_fn is not None
        # 评估曲线参数:每 eval_freq 训练步,用 deterministic policy 跑 n_eval 回合
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self._eval_env: Optional[VascularNavEnv] = None
        self.reward_params = dict(reward_params)
        self.max_episodes = int(max_episodes)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.seed = int(seed)
        self.speed_config = resolve_speed_config(speed_config)
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.stop_requested = False
        self.model = None
        self._env: Optional[VascularNavEnv] = None
        # 可视化节流(~25 fps),保证 UI 跟得上 + 不刷爆 signal queue
        self._viz_min_interval_s = 0.040
        self._viz_last_t = 0.0
        self._run_id = ""
        self._run_dir: Optional[Path] = None
        self._episode_csv_file = None
        self._episode_csv = None
        self._eval_csv_file = None
        self._eval_csv = None
        self._best_eval_score: Optional[tuple[float, ...]] = None
        self._best_model_path: Optional[Path] = None
        self._eval_success_streak = 0
        self._best_min_success_streak = 5

    def request_stop(self):
        self.stop_requested = True

    def _prepare_run_logs(self) -> tuple[str, Path, bool]:
        self._run_id = f"{self.algo_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self._run_dir = self.ckpt_dir / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        config_path = self._run_dir / "training_config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": self._run_id,
                    "algorithm": self.display_name,
                    "max_episodes": self.max_episodes,
                    "max_steps_per_episode": self.max_steps_per_episode,
                    "seed": self.seed,
                    "eval_freq": self.eval_freq,
                    "n_eval_episodes": self.n_eval_episodes,
                    "best_model_min_success_streak": self._best_min_success_streak,
                    "model_hyperparameters": self._model_hyperparameters(),
                    "reward_params": self.reward_params,
                    "guidewire_params": DEFAULT_GUIDEWIRE_PARAMS,
                    "speed_config": self.speed_config.to_dict(),
                    "resolved_centerline_bias": float(
                        MujocoGuidewireConfig()
                        .copy(**DEFAULT_GUIDEWIRE_PARAMS)
                        .centerline_bias
                    ),
                    "sdf_enabled": self.sdf_enabled,
                    "generalize": self.centerlines is not None,
                    "n_routes": len(self.centerlines) if self.centerlines is not None else 1,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        self._episode_csv_file = (self._run_dir / "episode_log.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._episode_csv = csv.DictWriter(
            self._episode_csv_file,
            fieldnames=[
                "episode", "timestep", "reward", "avg_reward_50", "success",
                "success_rate_50", "length", "blocked", "worst_sdf",
                "final_progress", "max_progress", "route_idx",
                "blocked_advance", "advance", "retract", "rotate_cw",
                "rotate_ccw", "proximal_advance", "distal_advance",
                "requested_advance_mm",
                *(f"reward_{key}" for key in REWARD_COMPONENT_KEYS),
            ],
        )
        self._episode_csv.writeheader()

        self._eval_csv_file = (self._run_dir / "eval_log.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._eval_csv = csv.DictWriter(
            self._eval_csv_file,
            fieldnames=[
                "timestep", "success_rate", "mean_reward", "mean_length",
                "mean_blocked", "mean_final_progress", "mean_max_progress", "n_eval",
                "evaluation_type",
            ],
        )
        self._eval_csv.writeheader()

        if importlib.util.find_spec("tensorboard") is None:
            self.status_changed.emit(
                "TensorBoard is not installed; only CSV logs will be saved"
            )
            return self._run_id, self._run_dir, False
        return self._run_id, self._run_dir, True

    def _write_episode_log(self, payload: dict):
        if self._episode_csv is None:
            return
        self._episode_csv.writerow({
            "episode": int(payload.get("index", 0)),
            "timestep": int(payload.get("timestep", 0)),
            "reward": float(payload.get("reward", 0.0)),
            "avg_reward_50": float(payload.get("avg_reward_50", 0.0)),
            "success": int(bool(payload.get("success", False))),
            "success_rate_50": float(payload.get("success_rate_50", 0.0)),
            "length": int(payload.get("length", 0)),
            "blocked": int(payload.get("blocked", 0)),
            "worst_sdf": float(payload.get("worst_sdf", 0.0)),
            "final_progress": float(payload.get("final_progress", 0.0)),
            "max_progress": float(payload.get("max_progress", 0.0)),
            "route_idx": int(payload.get("route_idx", 0)),
            "blocked_advance": int(payload.get("blocked_advance", 0)),
            "advance": int(payload.get("advance", 0)),
            "retract": int(payload.get("retract", 0)),
            "rotate_cw": int(payload.get("rotate_cw", 0)),
            "rotate_ccw": int(payload.get("rotate_ccw", 0)),
            "proximal_advance": int(payload.get("proximal_advance", 0)),
            "distal_advance": int(payload.get("distal_advance", 0)),
            "requested_advance_mm": float(payload.get("requested_advance_mm", 0.0)),
            **{
                f"reward_{key}": float(payload.get(f"reward_{key}", 0.0))
                for key in REWARD_COMPONENT_KEYS
            },
        })
        self._episode_csv_file.flush()

    def _write_eval_log(self, payload: dict):
        if self._eval_csv is None:
            return
        self._eval_csv.writerow({
            "timestep": int(payload.get("timestep", 0)),
            "success_rate": float(payload.get("success_rate", 0.0)),
            "mean_reward": float(payload.get("mean_reward", 0.0)),
            "mean_length": float(payload.get("mean_length", 0.0)),
            "mean_blocked": float(payload.get("mean_blocked", 0.0)),
            "mean_final_progress": float(payload.get("mean_final_progress", 0.0)),
            "mean_max_progress": float(payload.get("mean_max_progress", 0.0)),
            "n_eval": int(payload.get("n_eval", 0)),
            "evaluation_type": str(payload.get("evaluation_type", "periodic")),
        })
        self._eval_csv_file.flush()

    def _close_run_logs(self):
        for attr in ("_episode_csv_file", "_eval_csv_file"):
            f = getattr(self, attr, None)
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        self._episode_csv = None
        self._eval_csv = None

    def _plot_training_curve(self, ckpt: Path) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        script = Path(__file__).resolve().parents[2] / "scripts" / "plot_training_curve.py"
        if not script.exists():
            self.status_changed.emit(f"Training-curve script not found: {script}")
            return None, None, None

        png_path = ckpt.with_name(f"{ckpt.stem}_training_curve.png")
        csv_path = ckpt.with_name(f"{ckpt.stem}_training_curve.csv")
        loss_png_path = ckpt.with_name(f"{ckpt.stem}_diagnostics.png")
        cmd = [
            sys.executable,
            str(script),
            str(ckpt),
            "--output",
            str(png_path),
            "--csv",
            str(csv_path),
        ]
        try:
            subprocess.run(
                cmd,
                cwd=str(script.parents[1]),
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception as exc:
            self.status_changed.emit(
                f"Failed to generate training curves: {type(exc).__name__}: {exc}"
            )
            return None, None, None
        return png_path, csv_path, loss_png_path if loss_png_path.exists() else None

    def _run_and_emit_evaluation(
        self,
        model,
        timestep: int,
        evaluation_type: str = "periodic",
    ):
        """用 deterministic policy 在独立 eval env 上跑 n_eval 回合,统计均值后 emit。

        这是论文式"成功率评估曲线"的数据源:反映当前模型真实水平(无探索噪声)。
        """
        env = self._eval_env
        if env is None:
            return
        succ, rews, lens, blks, final_progs, max_progs = [], [], [], [], [], []
        for eval_index in range(self.n_eval_episodes):
            # Reuse the same deterministic evaluation seeds at every checkpoint
            # so algorithm/checkpoint comparisons see identical scenarios.
            obs, _ = env.reset(seed=self.seed + 100_000 + eval_index)
            done = False
            total_r = 0.0
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                total_r += float(r)
                done = bool(term or trunc)
                if self.stop_requested:
                    break
            succ.append(1.0 if info.get("success") else 0.0)
            rews.append(total_r)
            lens.append(int(info.get("ep_length", 0)))
            blks.append(int(info.get("ep_blocked", 0)))
            final_progs.append(float(info.get("progress", 0.0)))
            max_progs.append(float(info.get("best_progress", 0.0)))
            if self.stop_requested:
                break
        if not succ:
            return
        payload = {
            "timestep":     int(timestep),
            "success_rate": float(np.mean(succ)),
            "mean_reward":  float(np.mean(rews)),
            "mean_length":  float(np.mean(lens)),
            "mean_blocked": float(np.mean(blks)),
            "mean_final_progress": float(np.mean(final_progs)),
            "mean_max_progress": float(np.mean(max_progs)),
            "n_eval":       len(succ),
            "evaluation_type": str(evaluation_type),
        }
        self._write_eval_log(payload)
        self.eval_done.emit(payload)

        if payload["success_rate"] >= 1.0:
            self._eval_success_streak += 1
        else:
            self._eval_success_streak = 0

        # A single deterministic success can be caused by an initially lucky
        # action bias. Only promote a best model after sustained success.
        score = (
            payload["success_rate"],
            -payload["mean_blocked"],
            -payload["mean_length"],
            payload["mean_reward"],
        )
        eligible = self._eval_success_streak >= self._best_min_success_streak
        if eligible and (self._best_eval_score is None or score > self._best_eval_score):
            self._best_eval_score = score
            self._best_model_path = self._run_dir / "best_model.zip"
            model.save(str(self._best_model_path))
            self._write_eval_log({**payload, "evaluation_type": "best"})
            self.status_changed.emit(
                    f"Best evaluation model updated success={payload['success_rate']:.2f} "
                f"length={payload['mean_length']:.1f} "
                f"streak={self._eval_success_streak}"
            )
        return payload

    def _make_env(self):
        return self.env_class(
            centerlines=self.centerlines,
            sdf_fn=self.sdf_fn,
            reward_params=self.reward_params,
            speed_config=self.speed_config,
            max_steps=self.max_steps_per_episode,
        ) if self.centerlines is not None else self.env_class(
            centerline=self.centerline,
            sdf_fn=self.sdf_fn,
            reward_params=self.reward_params,
            speed_config=self.speed_config,
            max_steps=self.max_steps_per_episode,
        )

    def _model_hyperparameters(self) -> dict:
        return {
            "policy": "MlpPolicy",
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 4,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "seed": self.seed,
            "device": "cpu",
            "verbose": 0,
        }

    def _build_model(self, env):
        config = self._model_hyperparameters()
        policy = config.pop("policy")
        return PPO(policy, env, **config)

    def _on_env_step(self, polyline, cam_pose, info):
        """env.step_observer 回调(训练线程内)。节流后跨线程 emit 给 UI。"""
        now = time.monotonic()
        if now - self._viz_last_t < self._viz_min_interval_s:
            return
        self._viz_last_t = now
        try:
            # numpy 数组 copy 后跨线程安全;cam_pose dict 已是普通容器
            pl = polyline.copy() if polyline is not None else None
            self.step_observed.emit(pl, cam_pose)
        except Exception:
            pass

    def run(self):
        try:
            set_random_seed(self.seed)
            run_id, run_dir, tensorboard_enabled = self._prepare_run_logs()
            self.status_changed.emit(f"Log directory {run_dir}")
            self.status_changed.emit("Initializing environment …")
            if self.centerlines is not None:
                self.status_changed.emit(
                    f"Initializing environment "
                    f"({len(self.centerlines)} generalized routes) …")
            else:
                self.status_changed.emit("Initializing environment …")
            print("1")
            self._env = self._make_env()
            self._env.action_space.seed(self.seed)
            self._env.reset(seed=self.seed)
            print("2")
            # 把训练 env 的实时状态 hook 到 trainer signal,UI 主线程接收后渲染
            self._env.step_observer = self._on_env_step

            # 独立 eval env(deterministic 评估用,不挂 viz,不和训练 env 竞态)
            if self.eval_freq > 0:
                self.status_changed.emit("Initializing evaluation environment …")
                self._eval_env = self._make_env()
                self._eval_env.action_space.seed(self.seed + 100_000)
            print("3")

            self.status_changed.emit(f"Initializing {self.display_name} …")
            self.model = self._build_model(self._env)
            print("4")
            logger_formats = ["csv"]
            if tensorboard_enabled:
                logger_formats.append("tensorboard")
            self.model.set_logger(configure(str(run_dir), logger_formats))

            cb = _PPOCallback(self)
            # SB3 需要一个足够大的 timestep 上限；真正的停止条件由 callback
            # 按用户指定的 max_episodes 精确控制。
            total_ts = self.max_episodes * self.max_steps_per_episode
            # RL 面板状态栏只显示简短训练态;详细 ep/r/avg/succ 走 debug 面板
            self.status_changed.emit(
                f"Training (target {self.max_episodes} episodes, seed={self.seed})"
            )
            self.model.learn(total_timesteps=total_ts, callback=cb, tb_log_name=run_id)
            if self._eval_env is not None and not self.stop_requested:
                self.status_changed.emit(
                    "Training complete; evaluating the final model …"
                )
                self._run_and_emit_evaluation(
                    self.model,
                    int(self.model.num_timesteps),
                    evaluation_type="final",
                )
            print("5")
            # 保存
            ckpt = self.ckpt_dir / f"{run_id}.zip"
            self.model.save(str(ckpt))
            self.status_changed.emit(f"Saved {ckpt.name}")
            if self._best_model_path is not None:
                self.status_changed.emit(f"Best evaluation model {self._best_model_path}")
            png_path, _csv_path, loss_png_path = self._plot_training_curve(ckpt)
            if png_path is not None:
                self.status_changed.emit(f"Training curves generated: {png_path.name}")
            if loss_png_path is not None:
                self.status_changed.emit(
                    f"Training diagnostics generated: {loss_png_path.name}"
                )
            self.finished.emit(str(ckpt))
        except Exception as exc:
            traceback.print_exc()
            self.status_changed.emit(f"Training failed: {type(exc).__name__}: {exc}")
            self.finished.emit("")
        finally:
            self._close_run_logs()
            for e in (self._env, self._eval_env):
                try:
                    if e is not None:
                        e.close()
                except Exception:
                    pass


class SACTrainer(PPOTrainer):
    """SB3 SAC trainer running beside PPO on a continuous-action env."""

    algo_name = "sac"
    display_name = "SAC"
    env_class = VascularNavContinuousEnv

    def _model_hyperparameters(self) -> dict:
        return {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "policy_kwargs": {"net_arch": [64, 64]},
            "seed": self.seed,
            "verbose": 0,
        }

    def _build_model(self, env):
        config = self._model_hyperparameters()
        policy = config.pop("policy")
        return SAC(policy, env, **config)

# Stable overrides/new algorithms. These definitions intentionally appear after
# the original classes so imports use the tuned versions below.
class TD3Trainer(PPOTrainer):
    """SB3 TD3 trainer for continuous primitive selection."""

    algo_name = "td3"
    display_name = "TD3"
    env_class = VascularNavContinuousEnv

    def _model_hyperparameters(self) -> dict:
        return {
            "policy": "MlpPolicy",
            "learning_rate": 1e-4,
            "buffer_size": 200_000,
            "learning_starts": 5_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "policy_delay": 2,
            "target_policy_noise": 0.1,
            "target_noise_clip": 0.2,
            "action_noise_sigma": 0.20,
            "policy_kwargs": {"net_arch": [128, 128]},
            "seed": self.seed,
            "verbose": 0,
        }

    def _build_model(self, env):
        config = self._model_hyperparameters()
        policy = config.pop("policy")
        noise_sigma = float(config.pop("action_noise_sigma"))
        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[-1]),
            sigma=noise_sigma * np.ones(env.action_space.shape[-1]),
        )
        return TD3(policy, env, action_noise=action_noise, **config)


class DDPGTrainer(PPOTrainer):
    """SB3 DDPG trainer for continuous primitive selection."""

    algo_name = "ddpg"
    display_name = "DDPG"
    env_class = VascularNavContinuousEnv

    def _model_hyperparameters(self) -> dict:
        return {
            "policy": "MlpPolicy",
            "learning_rate": 1e-4,
            "buffer_size": 200_000,
            "learning_starts": 5_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "action_noise_sigma": 0.20,
            "policy_kwargs": {"net_arch": [128, 128]},
            "seed": self.seed,
            "verbose": 0,
        }

    def _build_model(self, env):
        config = self._model_hyperparameters()
        policy = config.pop("policy")
        noise_sigma = float(config.pop("action_noise_sigma"))
        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[-1]),
            sigma=noise_sigma * np.ones(env.action_space.shape[-1]),
        )
        return DDPG(policy, env, action_noise=action_noise, **config)


class DQNTrainer(PPOTrainer):
    """SB3 DQN trainer for the same discrete primitive actions as PPO."""

    algo_name = "dqn"
    display_name = "DQN"
    env_class = VascularNavEnv

    def _model_hyperparameters(self) -> dict:
        return {
            "policy": "MlpPolicy",
            "learning_rate": 1e-4,
            "buffer_size": 200_000,
            "learning_starts": 10_000,
            "batch_size": 128,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 5_000,
            "exploration_fraction": 0.35,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "policy_kwargs": {"net_arch": [128, 128]},
            "seed": self.seed,
            "verbose": 0,
        }

    def _build_model(self, env):
        config = self._model_hyperparameters()
        policy = config.pop("policy")
        return DQN(policy, env, **config)
