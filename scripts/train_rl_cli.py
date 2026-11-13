"""Command-line RL training entrypoint.

Examples:
    python scripts/train_rl_cli.py --algo PPO --episodes 500 --max-steps 1200
    python scripts/train_rl_cli.py --algo TD3 --episodes 800 --success-progress 0.99
    python scripts/train_rl_cli.py --algo DQN --generalize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from zmd.centerline.centerline_model import Centerline, pick_route
from zmd.rl.envs.vascular_env import DEFAULT_REWARD_PARAMS
from zmd.rl.speed_schedule import build_speed_config
from zmd.scene.scene_config import DEFAULT_ENTRY_POINT_PRESETS, DEFAULT_EXIT_POINT_PRESETS
from zmd.utils.paths import PROJECTS_DIR


TRAINERS = {
    "PPO": "PPOTrainer",
    "SAC": "SACTrainer",
    "TD3": "TD3Trainer",
    "DDPG": "DDPGTrainer",
    "DQN": "DQNTrainer",
}


def build_route_centerline(branches, entry_idx: int, exit_idx: int) -> Centerline:
    entry = DEFAULT_ENTRY_POINT_PRESETS[entry_idx]
    exit_ = DEFAULT_EXIT_POINT_PRESETS[exit_idx]
    branch = pick_route(
        branches,
        np.asarray(entry["entry_point"], dtype=float),
        np.asarray(exit_["exit_point"], dtype=float),
    )
    pts = np.asarray(branch.points, dtype=float)
    radii = branch.radius
    entry_point = np.asarray(entry["entry_point"], dtype=float)
    if np.linalg.norm(pts[0] - entry_point) > np.linalg.norm(pts[-1] - entry_point):
        pts = pts[::-1].copy()
        if radii is not None:
            radii = radii[::-1].copy()

    n = len(pts)
    entry_offset = max(0, min(2, n - 4))
    pullback_fraction = float(exit_.get("pullback_fraction", 0.10))
    exit_offset = max(0, min(int(round(n * pullback_fraction)), n - entry_offset - 2))
    trimmed = pts[entry_offset : n - exit_offset]
    trimmed_radii = radii[entry_offset : n - exit_offset] if radii is not None else None
    if len(trimmed) < 2:
        raise ValueError(f"Route {entry_idx}->{exit_idx} is too short after trimming")

    return Centerline(
        points=trimmed,
        radius=trimmed_radii,
        source_point=trimmed[0].copy(),
        target_point=trimmed[-1].copy(),
        source="vmtk",
        branches=branches,
    )


def build_centerlines(branches, generalize: bool, entry_idx: int, exit_idx: int):
    if not generalize:
        return build_route_centerline(branches, entry_idx, exit_idx), None
    routes = []
    for ei in range(len(DEFAULT_ENTRY_POINT_PRESETS)):
        for xi in range(len(DEFAULT_EXIT_POINT_PRESETS)):
            routes.append(build_route_centerline(branches, ei, xi))
    return None, routes


def load_vessel_sdf():
    """Load the same vessel SDF used by the Qt workspace training path."""
    try:
        import pyvista as pv
        import vtk

        stl = (PROJECTS_DIR / "cache" / "VascuModel_collision.stl").resolve()
        if not stl.exists():
            print(f"[sdf] missing {stl}; training without SDF", flush=True)
            return None
        mesh = pv.read(str(stl))
        implicit = vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(mesh)

        def _sdf(p, _f=implicit):
            try:
                return float(_f.EvaluateFunction([float(p[0]), float(p[1]), float(p[2])]))
            except Exception:
                return 0.0

        _ = _sdf([0.0, 0.0, 0.0])
        print(f"[sdf] loaded {stl} ({mesh.n_cells} tris)", flush=True)
        return _sdf
    except Exception as exc:
        print(f"[sdf] load failed: {type(exc).__name__}: {exc}; training without SDF", flush=True)
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL policy without opening the Qt UI.")
    parser.add_argument("--algo", choices=sorted(TRAINERS), default="PPO")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--entry", type=int, default=0, help="Entry preset index, 0-based.")
    parser.add_argument("--exit", type=int, default=2, help="Exit preset index, 0-based.")
    parser.add_argument("--generalize", action="store_true", help="Train on all entry/exit routes.")
    parser.add_argument("--success-progress", type=float, default=0.92)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=PROJECTS_DIR / "cache" / "rl")
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--speed-profile",
        type=str.upper,
        choices=(
            "B0", "A1", "A2", "A3", "A4", "A5",
            "A6", "A7", "P75", "P85", "P90", "P95", "CUSTOM",
        ),
        default="B0",
        help=(
            "Forward-step condition: B0=fixed 0.50 mm; "
            "A1-A7/P75/P85/P90/P95=segmented presets."
        ),
    )
    parser.add_argument(
        "--proximal-push-mm", type=float,
        help="Custom forward step before the progress switch (mm).",
    )
    parser.add_argument(
        "--distal-push-mm", type=float,
        help="Custom forward step at/after the progress switch (mm).",
    )
    parser.add_argument(
        "--switch-progress", type=float,
        help="Custom normalized centerline arc-length switch point, e.g. 0.80.",
    )
    parser.add_argument("--no-sdf", action="store_true", help="Disable vessel SDF collision checks.")
    return parser.parse_args()


def main():
    args = parse_args()
    vtp_path = PROJECTS_DIR / "cache" / "VascuModel_centerline.vtp"
    branches = Centerline.load_branches_from_vtp(vtp_path)
    if not branches:
        raise FileNotFoundError(
            f"No centerline branches found in {vtp_path}. Run scripts/vmtk_extract_centerline.py first."
        )

    entry_idx = max(0, min(args.entry, len(DEFAULT_ENTRY_POINT_PRESETS) - 1))
    exit_idx = max(0, min(args.exit, len(DEFAULT_EXIT_POINT_PRESETS) - 1))
    centerline, centerlines = build_centerlines(branches, args.generalize, entry_idx, exit_idx)

    import zmd.rl.trainer as trainer_module

    trainer_cls = getattr(trainer_module, TRAINERS[args.algo])
    reward_params = dict(DEFAULT_REWARD_PARAMS)
    reward_params["success_progress"] = float(args.success_progress)
    speed_config = build_speed_config(
        args.speed_profile,
        proximal_push_mm=args.proximal_push_mm,
        distal_push_mm=args.distal_push_mm,
        switch_progress=args.switch_progress,
    )
    if args.generalize:
        route_name = "all_routes"
    else:
        route_name = f"entry{entry_idx + 1}_exit{exit_idx + 1}"
    ckpt_dir = Path(args.output_root) / args.algo.lower() / route_name / f"repeat_{args.repeat_index:02d}"
    sdf_fn = None if args.no_sdf else load_vessel_sdf()
    trainer = trainer_cls(
        centerline=centerline,
        centerlines=centerlines,
        sdf_fn=sdf_fn,
        reward_params=reward_params,
        max_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        ckpt_dir=str(ckpt_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        seed=args.seed,
        speed_config=speed_config,
    )
    print(
        f"[config] algo={args.algo} route={route_name} repeat={args.repeat_index} "
        f"seed={args.seed} episodes={args.episodes} max_steps={args.max_steps} "
        f"speed={speed_config.profile} "
        f"push={speed_config.proximal_push_mm:.2f}/{speed_config.distal_push_mm:.2f}mm "
        f"switch={speed_config.switch_progress:.2f} "
        f"sdf={sdf_fn is not None} ckpt_dir={ckpt_dir}",
        flush=True,
    )
    trainer.status_changed.connect(lambda text: print(f"[status] {text}", flush=True))
    trainer.episode_ended.connect(
        lambda ep: print(
            "[episode] "
            f"#{ep['index']} step={ep['timestep']} reward={ep['reward']:.2f} "
            f"success={int(ep['success'])} sr50={ep['success_rate_50']:.2f}",
            flush=True,
        )
    )
    trainer.eval_done.connect(
        lambda ev: print(
            "[eval] "
            f"step={ev['timestep']} success_rate={ev['success_rate']:.2f} "
            f"mean_reward={ev['mean_reward']:.2f}",
            flush=True,
        )
    )
    trainer.finished.connect(lambda path: print(f"[finished] {path}", flush=True))
    trainer.run()


if __name__ == "__main__":
    main()
