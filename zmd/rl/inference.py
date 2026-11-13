"""RL inference replay controllers for the Qt workspace.

Every controller uses the same observation and primitive-action mapping as its
training environment.  A shared distance schedule changes only the ADVANCE
step, while a recorder saves step-level and one-row summary CSV files.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3

from zmd.rl.envs.vascular_env import (
    ACTION_ADVANCE,
    ACTION_NAMES,
    ACTION_NOOP,
    ACTION_RETRACT,
    ACTION_ROTATE_CCW,
    ACTION_ROTATE_CW,
    build_observation,
)
from zmd.rl.inference_metrics import (
    load_checkpoint_training_config,
    make_step_row,
    summarize_rollout,
    write_inference_csvs,
)
from zmd.rl.speed_schedule import resolve_speed_config


class PPOInferenceController(QObject):
    """Load one policy and drive a UI ``MujocoGuidewire`` to termination."""

    status_changed = pyqtSignal(str)
    step_taken = pyqtSignal(int)
    finished = pyqtSignal(bool)
    metrics_saved = pyqtSignal(str, str)

    MODEL_CLASS = PPO
    DISPLAY_NAME = "PPO"
    CONTINUOUS_ACTION = False
    ACTION_NAMES = ACTION_NAMES

    def __init__(
        self,
        ckpt_path: str,
        guidewire,
        parent=None,
        tick_ms: int = 90,
        max_steps: int = 2000,
        success_progress: float = 0.99,
        speed_config=None,
        output_dir: Optional[str] = None,
        run_metadata: Optional[dict] = None,
    ):
        super().__init__(parent)
        self.ckpt_path = str(ckpt_path)
        self.guidewire = guidewire
        self.model = None
        self._max_steps = int(max_steps)
        self._success_progress = float(success_progress)
        self.speed_config = resolve_speed_config(speed_config)
        self.output_dir = Path(output_dir) if output_dir else Path(self.ckpt_path).parent
        self.run_metadata = dict(run_metadata or {})
        self.run_metadata.setdefault("algorithm", self.DISPLAY_NAME)
        self.run_metadata.setdefault("checkpoint", Path(self.ckpt_path).name)
        self.run_metadata.setdefault("deterministic", 1)
        training_config = load_checkpoint_training_config(Path(self.ckpt_path))
        trained_speed = training_config.get("speed_config", {})
        if not isinstance(trained_speed, dict):
            trained_speed = {}
        trained_rewards = training_config.get("reward_params", {})
        if not isinstance(trained_rewards, dict):
            trained_rewards = {}
        self.run_metadata.setdefault("seed", training_config.get("seed", ""))
        self.run_metadata.setdefault(
            "training_speed_profile", trained_speed.get("profile", "B0_LEGACY")
        )
        self.run_metadata.setdefault(
            "training_proximal_push_mm", trained_speed.get("proximal_push_mm", 0.50)
        )
        self.run_metadata.setdefault(
            "training_distal_push_mm", trained_speed.get("distal_push_mm", 0.50)
        )
        self.run_metadata.setdefault(
            "training_switch_progress", trained_speed.get("switch_progress", "")
        )
        self.run_metadata.setdefault(
            "training_success_progress", trained_rewards.get("success_progress", "")
        )
        self._tick_ms = int(tick_ms)

        self._step_count = 0
        self._best_progress = 0.0
        self._distal_speed_active = False
        self._rows: list[dict] = []
        self._finalized = False
        self.last_rollout_path: Optional[Path] = None
        self.last_summary_path: Optional[Path] = None

        self.timer = QTimer(self)
        self.timer.setInterval(self._tick_ms)
        self.timer.timeout.connect(self._tick)

        cl_pts = np.asarray(guidewire.centerline.points, dtype=float)
        self._bbox_center = (cl_pts.min(0) + cl_pts.max(0)) * 0.5
        self._bbox_extent = np.maximum(cl_pts.max(0) - cl_pts.min(0), 1e-3)
        self._travel_length = float(getattr(guidewire.backend, "_travel_length", 1.0))
        self._max_length = float(getattr(guidewire.backend, "_max_length", 1.0))

    def load(self) -> bool:
        try:
            self.model = self.MODEL_CLASS.load(self.ckpt_path)
            self.status_changed.emit(
                f"{self.DISPLAY_NAME} model loaded: {Path(self.ckpt_path).name}"
            )
            return True
        except Exception as exc:
            self.status_changed.emit(
                f"{self.DISPLAY_NAME} load failed: {type(exc).__name__}: {exc}"
            )
            self.model = None
            return False

    def start(self):
        if self.model is None:
            self.status_changed.emit("No model loaded")
            return
        self._step_count = 0
        self._best_progress = 0.0
        self._distal_speed_active = False
        self._rows = []
        self._finalized = False
        self.last_rollout_path = None
        self.last_summary_path = None
        try:
            self.guidewire.reset()
        except Exception:
            pass
        cfg = self.speed_config
        self.status_changed.emit(
            f"Inference replay {cfg.profile}: "
            f"{cfg.proximal_push_mm:.2f}/{cfg.distal_push_mm:.2f} mm, "
            f"switch={cfg.switch_progress:.2f}"
        )
        self.timer.start()

    def stop(self):
        if self.timer.isActive():
            self.timer.stop()
            self._finalize(False, "user_stopped", emit_finished=False)
        self.status_changed.emit(f"Inference stopped (step={self._step_count})")

    def is_running(self) -> bool:
        return self.timer.isActive()

    def _predict_primitive(self, obs) -> tuple[int, object]:
        action_raw, _ = self.model.predict(obs, deterministic=True)
        action = int(np.asarray(action_raw).reshape(-1)[0])
        return action, action

    def _tick(self):
        try:
            obs = build_observation(
                self.guidewire,
                self._bbox_center,
                self._bbox_extent,
                self._max_length,
                self._travel_length,
            )
            action, raw_action = self._predict_primitive(obs)
            applied, requested_push_mm, speed_phase = self._apply_action(action)
            self._step_count += 1

            stats = dict(self.guidewire.backend.last_constraint_stats)
            progress = float(stats.get("progress", 0.0))
            stats["success"] = bool(progress >= self._success_progress)
            self._best_progress = max(self._best_progress, progress)
            self._rows.append(make_step_row(
                step=self._step_count,
                action_index=action,
                raw_action=raw_action,
                action_applied=applied,
                requested_push_mm=requested_push_mm,
                speed_profile=self.speed_config.profile,
                speed_phase=speed_phase,
                stats=stats,
                guidewire=self.guidewire,
                max_progress=self._best_progress,
                metadata=self.run_metadata,
            ))
            self.step_taken.emit(action)

            if progress >= self._success_progress:
                self.timer.stop()
                self._finalize(True, "success")
            elif self._step_count >= self._max_steps:
                self.timer.stop()
                self._finalize(False, "max_steps")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.timer.stop()
            self.status_changed.emit(f"{self.DISPLAY_NAME} inference error: {exc}")
            self._finalize(False, f"exception:{type(exc).__name__}")

    def _apply_action(self, action: int) -> tuple[bool, float, str]:
        speed_phase = self.speed_config.phase(
            self._best_progress,
            self._distal_speed_active,
        )
        if speed_phase == "distal":
            self._distal_speed_active = True
        requested_push_mm = 0.0
        if action == ACTION_ADVANCE:
            requested_push_mm = self.speed_config.push_step_mm(
                self._best_progress,
                self._distal_speed_active,
            )
            applied = bool(self.guidewire.advance_by(requested_push_mm))
        elif action == ACTION_RETRACT:
            applied = bool(self.guidewire.retract_step())
        elif action == ACTION_ROTATE_CW:
            applied = bool(self.guidewire.rotate_clockwise())
        elif action == ACTION_ROTATE_CCW:
            applied = bool(self.guidewire.rotate_counterclockwise())
        elif action == ACTION_NOOP:
            applied = True
        else:
            applied = False
        return applied, requested_push_mm, speed_phase

    def _finalize(
        self,
        success: bool,
        termination_reason: str,
        *,
        emit_finished: bool = True,
    ):
        if self._finalized:
            return
        self._finalized = True
        cfg = self.speed_config
        summary_metadata = {
            **self.run_metadata,
            "speed_profile": cfg.profile,
            "proximal_push_mm": cfg.proximal_push_mm,
            "distal_push_mm": cfg.distal_push_mm,
            "switch_progress": cfg.switch_progress,
            "success_progress": self._success_progress,
            "tick_ms": self._tick_ms,
            "sdf_enabled": int(
                getattr(self.guidewire.backend, "_sdf_fn", None) is not None
            ),
        }
        summary = summarize_rollout(
            self._rows,
            success=success,
            termination_reason=termination_reason,
            metadata=summary_metadata,
            tick_ms=self._tick_ms,
        )
        route = str(self.run_metadata.get("route", "route"))
        safe_route = re.sub(r"[^0-9A-Za-z_-]+", "_", route).strip("_") or "route"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rollout_path = self.output_dir / (
            f"{Path(self.ckpt_path).stem}_infer_{safe_route}_"
            f"{cfg.profile.lower()}_{stamp}.csv"
        )
        try:
            self.last_rollout_path, self.last_summary_path = write_inference_csvs(
                self._rows,
                summary,
                rollout_path,
            )
            self.metrics_saved.emit(
                str(self.last_rollout_path),
                str(self.last_summary_path),
            )
        except Exception as exc:
            self.status_changed.emit(
                f"Failed to save inference metrics: {type(exc).__name__}: {exc}"
            )

        rotations = int(summary.get("rotation_count", 0))
        final_progress = float(summary.get("final_progress", 0.0))
        if success:
            self.status_changed.emit(
                f"Destination reached ✓ (step={self._step_count}, rotation={rotations})"
            )
        elif termination_reason == "max_steps":
            self.status_changed.emit(
                f"Step limit reached before destination (step={self._step_count}, "
                f"progress={final_progress * 100:.1f}%, rotation={rotations})"
            )
        if emit_finished:
            self.finished.emit(bool(success))


class SACInferenceController(PPOInferenceController):
    """SAC replay with continuous actions mapped to one primitive per tick."""

    MODEL_CLASS = SAC
    DISPLAY_NAME = "SAC"
    CONTINUOUS_ACTION = True

    def _predict_primitive(self, obs) -> tuple[int, object]:
        action_raw, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action_raw, dtype=np.float32).reshape(-1)
        slide = float(np.clip(action[0] if len(action) >= 1 else 0.0, -1.0, 1.0))
        rotate = float(np.clip(action[1] if len(action) >= 2 else 0.0, -1.0, 1.0))
        if abs(slide) >= abs(rotate):
            primitive = ACTION_ADVANCE if slide >= 0.0 else ACTION_RETRACT
        else:
            primitive = ACTION_ROTATE_CW if rotate >= 0.0 else ACTION_ROTATE_CCW
        return primitive, action.tolist()


class TD3InferenceController(SACInferenceController):
    MODEL_CLASS = TD3
    DISPLAY_NAME = "TD3"


class DDPGInferenceController(SACInferenceController):
    MODEL_CLASS = DDPG
    DISPLAY_NAME = "DDPG"


class DQNInferenceController(PPOInferenceController):
    MODEL_CLASS = DQN
    DISPLAY_NAME = "DQN"
