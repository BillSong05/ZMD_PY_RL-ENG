"""Command-line RL inference/evaluation without opening the Qt UI."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3

from scripts.train_rl_cli import (
    DEFAULT_REWARD_PARAMS,
    build_route_centerline,
    load_vessel_sdf,
)
from zmd.centerline.centerline_model import Centerline
from zmd.rl.envs.vascular_env import VascularNavContinuousEnv, VascularNavEnv
from zmd.rl.inference_metrics import (
    load_checkpoint_training_config,
    make_step_row,
    summarize_rollout,
    write_inference_csvs,
)
from zmd.rl.speed_schedule import build_speed_config
from zmd.scene.scene_config import DEFAULT_ENTRY_POINT_PRESETS, DEFAULT_EXIT_POINT_PRESETS
from zmd.utils.paths import PROJECTS_DIR


MODEL_CLASSES = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
    "DQN": DQN,
}


def infer_algo_from_path(path: Path) -> str:
    prefix = path.stem.split("_", 1)[0].upper()
    if prefix in MODEL_CLASSES:
        return prefix
    raise ValueError(f"Cannot infer algorithm from checkpoint name: {path.name}. Use --algo.")


def load_training_config(checkpoint: Path) -> dict:
    return load_checkpoint_training_config(checkpoint)


def load_training_reward_params(checkpoint: Path) -> dict:
    params = load_training_config(checkpoint).get("reward_params", {})
    return params if isinstance(params, dict) else {}


def parse_args():
    parser = argparse.ArgumentParser(description="Run one trained policy without the Qt UI.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--algo", choices=sorted(MODEL_CLASSES))
    parser.add_argument("--entry", type=int, default=0, help="Entry preset index, 0-based.")
    parser.add_argument("--exit", type=int, default=2, help="Exit preset index, 0-based.")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--success-progress",
        type=float,
        default=0.99,
        help="Inference success threshold. Defaults to 0.99, independent of checkpoint training_config.",
    )
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--output", type=Path, help="Optional rollout CSV path.")
    parser.add_argument("--summary-output", type=Path, help="Optional one-row summary CSV path.")
    parser.add_argument(
        "--speed-profile",
        type=str.upper,
        choices=(
            "B0", "A1", "A2", "A3", "A4", "A5",
            "A6", "A7", "P75", "P85", "P90", "P95", "CUSTOM",
        ),
        default="B0",
        help="Forward-step condition used only during this inference run.",
    )
    parser.add_argument("--proximal-push-mm", type=float)
    parser.add_argument("--distal-push-mm", type=float)
    parser.add_argument("--switch-progress", type=float)
    parser.add_argument(
        "--tick-ms", type=int, default=90,
        help="Assumed control period for estimated execution time; CLI does not sleep.",
    )
    parser.add_argument("--no-sdf", action="store_true", help="Disable vessel SDF collision checks.")
    return parser.parse_args()


def main():
    args = parse_args()
    algo = args.algo or infer_algo_from_path(args.checkpoint)
    model_cls = MODEL_CLASSES[algo]

    vtp_path = PROJECTS_DIR / "cache" / "VascuModel_centerline.vtp"
    branches = Centerline.load_branches_from_vtp(vtp_path)
    if not branches:
        raise FileNotFoundError(
            f"No centerline branches found in {vtp_path}. Run scripts/vmtk_extract_centerline.py first."
        )

    entry_idx = max(0, min(args.entry, len(DEFAULT_ENTRY_POINT_PRESETS) - 1))
    exit_idx = max(0, min(args.exit, len(DEFAULT_EXIT_POINT_PRESETS) - 1))
    centerline = build_route_centerline(branches, entry_idx, exit_idx)
    training_config = load_training_config(args.checkpoint)
    reward_params = {**DEFAULT_REWARD_PARAMS, **load_training_reward_params(args.checkpoint)}
    reward_params["success_progress"] = float(args.success_progress)
    speed_config = build_speed_config(
        args.speed_profile,
        proximal_push_mm=args.proximal_push_mm,
        distal_push_mm=args.distal_push_mm,
        switch_progress=args.switch_progress,
    )
    env_cls = VascularNavContinuousEnv if algo in {"SAC", "TD3", "DDPG"} else VascularNavEnv
    sdf_fn = None if args.no_sdf else load_vessel_sdf()
    env = env_cls(
        centerline=centerline,
        sdf_fn=sdf_fn,
        reward_params=reward_params,
        speed_config=speed_config,
        max_steps=int(args.max_steps),
    )
    model = model_cls.load(str(args.checkpoint))

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    rows = []
    info = {}
    step = 0
    repeat_match = re.search(r"repeat_(\d+)", str(args.checkpoint.parent))
    trained_speed = training_config.get("speed_config", {})
    if not isinstance(trained_speed, dict):
        trained_speed = {}
    trained_rewards = training_config.get("reward_params", {})
    if not isinstance(trained_rewards, dict):
        trained_rewards = {}
    metadata = {
        "algorithm": algo,
        "checkpoint": args.checkpoint.name,
        "entry": entry_idx + 1,
        "exit": exit_idx + 1,
        "route": f"entry{entry_idx + 1}_exit{exit_idx + 1}",
        "seed": training_config.get("seed", ""),
        "repeat_index": int(repeat_match.group(1)) if repeat_match else "",
        "training_speed_profile": trained_speed.get("profile", "B0_LEGACY"),
        "training_proximal_push_mm": trained_speed.get("proximal_push_mm", 0.50),
        "training_distal_push_mm": trained_speed.get("distal_push_mm", 0.50),
        "training_switch_progress": trained_speed.get("switch_progress", ""),
        "training_success_progress": trained_rewards.get("success_progress", ""),
        "deterministic": int(bool(args.deterministic)),
        "sdf_enabled": int(sdf_fn is not None),
    }
    while not done:
        action, _ = model.predict(obs, deterministic=bool(args.deterministic))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step += 1
        raw_action = action.tolist() if hasattr(action, "tolist") else action
        rows.append(make_step_row(
            step=step,
            action_index=int(info.get("action_index", -1)),
            raw_action=raw_action,
            action_applied=bool(info.get("action_applied", False)),
            requested_push_mm=float(info.get("requested_push_mm", 0.0)),
            speed_profile=str(info.get("speed_profile", speed_config.profile)),
            speed_phase=str(info.get("speed_phase", "proximal")),
            stats=info,
            guidewire=env.gw,
            max_progress=float(info.get("best_progress", 0.0)),
            reward=float(reward),
            total_reward=float(total_reward),
            metadata=metadata,
        ))
        done = bool(terminated or truncated)

    success = bool(info.get("success", False))
    progress = float(info.get("progress", 0.0))
    termination_reason = "success" if success else "max_steps"
    summary = summarize_rollout(
        rows,
        success=success,
        termination_reason=termination_reason,
        metadata={
            **metadata,
            "speed_profile": speed_config.profile,
            "proximal_push_mm": speed_config.proximal_push_mm,
            "distal_push_mm": speed_config.distal_push_mm,
            "switch_progress": speed_config.switch_progress,
            "success_progress": float(args.success_progress),
            "tick_ms": int(args.tick_ms),
        },
        tick_ms=int(args.tick_ms),
    )
    print(
        f"[result] algo={algo} entry={entry_idx + 1} exit={exit_idx + 1} "
        f"success={int(success)} progress={progress:.4f} steps={step} "
        f"total_reward={total_reward:.2f} speed={speed_config.profile} "
        f"advance={summary['advance_count']} retract={summary['retract_count']} "
        f"rotate={summary['rotation_count']} blocked_rate={summary['blocked_advance_rate']:.4f}",
        flush=True,
    )

    output = args.output
    if output is None:
        output = args.checkpoint.with_name(
            f"{args.checkpoint.stem}_infer_entry{entry_idx + 1}_exit{exit_idx + 1}_"
            f"{speed_config.profile.lower()}.csv"
        )
    rollout_path, summary_path = write_inference_csvs(
        rows,
        summary,
        output,
        args.summary_output,
    )
    env.close()
    print(f"[rollout] {rollout_path}", flush=True)
    print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
