from __future__ import annotations

import argparse
import base64
import csv
import json
import pickle
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator


FONT_FAMILY = "Times New Roman"
FIGURE_DPI = 300
_STYLE_CONFIGURED = False


def configure_plot_style() -> Path:
    """Apply one readable, publication-friendly style to every output figure."""
    global _STYLE_CONFIGURED
    try:
        font_path = Path(
            font_manager.findfont(FONT_FAMILY, fallback_to_default=False)
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Required font '{FONT_FAMILY}' is not installed; refusing to render "
            "with a fallback font."
        ) from exc

    if not _STYLE_CONFIGURED:
        mpl.rcParams.update(
            {
                "font.family": FONT_FAMILY,
                "font.size": 12,
                "axes.linewidth": 1.0,
                "axes.labelsize": 13,
                "axes.titlesize": 13,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 11,
                "figure.titlesize": 15,
                "mathtext.fontset": "stix",
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.dpi": FIGURE_DPI,
            }
        )
        _STYLE_CONFIGURED = True
    return font_path


def infer_algo_name(model_zip: Path) -> str:
    prefix = model_zip.stem.split("_", 1)[0].upper()
    return prefix if prefix else "RL"


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = float(np.mean(values[start : i + 1]))
    return out


def load_episode_info(model_zip: Path) -> list[dict]:
    with zipfile.ZipFile(model_zip) as zf:
        data = json.loads(zf.read("data"))

    buffer_info = data.get("ep_info_buffer", {})
    encoded = buffer_info.get(":serialized:")
    if not encoded:
        return []
    return list(pickle.loads(base64.b64decode(encoded)))


def load_episode_log(model_zip: Path) -> list[dict]:
    episode_log = model_zip.with_suffix("") / "episode_log.csv"
    if not episode_log.exists():
        return []
    episodes = []
    with episode_log.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(
                {
                    "episode": int(float(row.get("episode", len(episodes) + 1))),
                    "timestep": int(float(row.get("timestep", 0))),
                    "r": float(row.get("reward", 0.0)),
                    "l": int(float(row.get("length", 0))),
                    "reward_ma": float(row.get("avg_reward_50", 0.0)),
                    "success": float(row.get("success", 0.0)),
                    "success_rate": float(row.get("success_rate_50", 0.0)),
                    "t": 0.0,
                }
            )
    return episodes


def write_csv(path: Path, episodes: list[dict], x_values: np.ndarray, reward_ma: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode", "timestep", "reward", "reward_ma", "length",
                "success", "success_rate", "wall_time_s",
            ],
        )
        writer.writeheader()
        for i, ep in enumerate(episodes):
            writer.writerow(
                {
                    "episode": int(ep.get("episode", i + 1)),
                    "timestep": int(x_values[i]),
                    "reward": float(ep.get("r", 0.0)),
                    "reward_ma": float(reward_ma[i]),
                    "length": int(ep.get("l", 0)),
                    "success": int(float(ep.get("success", 0.0))),
                    "success_rate": float(ep.get("success_rate", 0.0)),
                    "wall_time_s": float(ep.get("t", 0.0)),
                }
            )


def load_progress_csv(model_zip: Path) -> tuple[list[str], list[dict]]:
    run_dir = model_zip.with_suffix("")
    progress_path = run_dir / "progress.csv"
    if not progress_path.exists():
        return [], []
    with progress_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def plot_loss_curve(model_zip: Path, output: Path) -> Path | None:
    configure_plot_style()
    fields, rows = load_progress_csv(model_zip)
    if not rows:
        return None

    algo = infer_algo_name(model_zip)
    metric_candidates = {
        "PPO": [
            "train/value_loss",
            "train/policy_gradient_loss",
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/explained_variance",
        ],
        "SAC": [
            "train/critic_loss",
            "train/actor_loss",
            "train/ent_coef",
            "train/ent_coef_loss",
        ],
        "TD3": ["train/critic_loss", "train/actor_loss"],
        "DDPG": ["train/critic_loss", "train/actor_loss"],
        "DQN": ["train/loss", "rollout/exploration_rate"],
    }
    metrics = [key for key in metric_candidates.get(algo, []) if key in fields]
    if not metrics:
        metrics = [
            key for key in ("train/critic_loss", "train/value_loss", "train/loss")
            if key in fields
        ]
    if not metrics:
        return None

    x_key = "time/total_timesteps" if "time/total_timesteps" in fields else None
    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), dpi=FIGURE_DPI,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for ax, metric in zip(axes.flat, metrics):
        xs, ys = [], []
        for i, row in enumerate(rows):
            try:
                y = float(row.get(metric, ""))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(y):
                continue
            if x_key:
                try:
                    x = float(row.get(x_key, i + 1))
                except (TypeError, ValueError):
                    x = float(i + 1)
            else:
                x = float(i + 1)
            xs.append(x)
            ys.append(y)

        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)
        if len(ys_arr):
            ma_window = min(10, max(1, len(ys_arr) // 4))
            ma = moving_average(ys_arr, window=ma_window)
            ax.plot(xs_arr, ys_arr, color="#f59e0b", lw=1.35, alpha=0.50, label="Raw")
            ax.plot(xs_arr, ma, color="#7c3aed", lw=2.0, label=f"Moving avg ({ma_window})")
        label = metric.replace("train/", "").replace("rollout/", "")
        ax.set_xlabel("Timestep" if x_key else "Log row")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(True, color="#d8d8d8", lw=0.55, alpha=0.7)
        ax.legend(loc="best", frameon=True, facecolor="#eef4ff", edgecolor="#9eb6e8")

    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(f"{algo} training diagnostics", fontsize=15)
    fig.tight_layout(pad=0.9, rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
    )
    plt.close(fig)
    return output


def plot_curve(model_zip: Path, output: Path, csv_output: Path | None) -> Path | None:
    configure_plot_style()
    # episode_log.csv contains the complete run. SB3's serialized
    # ep_info_buffer is only a rolling buffer of the latest 100 episodes and is
    # therefore only a fallback for legacy checkpoints without a CSV log.
    episodes = load_episode_log(model_zip)
    source_label = "full episode_log.csv"
    if not episodes:
        episodes = load_episode_info(model_zip)
        source_label = "latest SB3 ep_info_buffer"
    if not episodes:
        raise RuntimeError(f"No episode info was found in {model_zip}")

    rewards = np.array([float(ep.get("r", 0.0)) for ep in episodes], dtype=float)
    lengths = np.array([int(ep.get("l", 0)) for ep in episodes], dtype=float)
    wall_time = np.array([float(ep.get("t", 0.0)) for ep in episodes], dtype=float)
    logged_timesteps = np.array(
        [int(ep.get("timestep", 0)) for ep in episodes], dtype=float
    )
    timesteps = (
        logged_timesteps
        if len(logged_timesteps) and np.all(logged_timesteps > 0)
        else np.cumsum(lengths)
    )
    smooth_window = min(50, max(1, len(rewards)))
    reward_ma = moving_average(rewards, window=smooth_window)
    length_ma = moving_average(lengths, window=smooth_window)
    has_success = all("success" in ep for ep in episodes)
    successes = np.array(
        [float(ep.get("success", 0.0)) for ep in episodes], dtype=float
    )
    success_ma = moving_average(successes, window=smooth_window)

    if csv_output is not None:
        write_csv(csv_output, episodes, timesteps, reward_ma)

    n_axes = 3 if has_success else 2
    fig, axes = plt.subplots(
        1,
        n_axes,
        figsize=(16.8 if has_success else 11.4, 4.4),
        dpi=FIGURE_DPI,
    )
    fig.patch.set_facecolor("white")

    blue = "#2339ff"
    red = "#ed3426"
    green = "#62e83f"
    pale_blue = "#9ec9f5"

    ax = axes[0]
    ax.plot(timesteps, rewards, color=pale_blue, lw=2.0, alpha=0.65, label="Episode")
    ax.plot(timesteps, reward_ma, color=blue, lw=2.4, label="Moving avg")
    ax.axhline(0, color="#555555", lw=0.8, alpha=0.45)
    ax.scatter(timesteps, rewards, s=16, color=blue, alpha=0.75, zorder=3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, color="#d8d8d8", lw=0.55, alpha=0.7)
    ax.legend(loc="lower right", frameon=True, facecolor="#eef4ff", edgecolor="#9eb6e8")
    algo = infer_algo_name(model_zip)
    ax.text(
        0.04,
        0.08,
        f"{algo} run",
        transform=ax.transAxes,
        fontsize=12.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#eef4ed", edgecolor="#c7d8c3", alpha=0.95),
    )

    ax = axes[1]
    ax.plot(timesteps, lengths, color=red, lw=2.1, label="Episode length")
    ax.plot(timesteps, length_ma, color=green, lw=2.0, label=f"Length avg ({smooth_window})")
    ax.fill_between(timesteps, lengths, np.min(lengths), color=red, alpha=0.10)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Length")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, color="#d8d8d8", lw=0.55, alpha=0.7)
    ax.legend(loc="lower left", frameon=True, facecolor="#eef4ff", edgecolor="#9eb6e8")
    ax.text(
        0.72,
        0.82,
        f"{len(episodes)} episodes",
        transform=ax.transAxes,
        fontsize=12.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#eef4ed", edgecolor="#c7d8c3", alpha=0.95),
    )

    if has_success:
        ax = axes[2]
        ax.scatter(timesteps, successes, s=9, color="#94a3b8", alpha=0.35, label="Episode success")
        ax.plot(timesteps, success_ma, color="#16a34a", lw=2.3, label=f"Success rate ({smooth_window})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Success rate")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, color="#d8d8d8", lw=0.55, alpha=0.7)
        ax.legend(loc="lower right", frameon=True, facecolor="#eef4ff", edgecolor="#9eb6e8")

    title = model_zip.stem
    subtitle = f"{source_label}, {len(episodes)} episodes, {int(timesteps[-1])} timesteps"
    fig.suptitle(f"{title} training curve\n{subtitle}", fontsize=14, y=1.03)
    fig.tight_layout(pad=0.9)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
    )
    plt.close(fig)
    diagnostics_output = output.with_name(f"{model_zip.stem}_diagnostics.png")
    return plot_loss_curve(model_zip, diagnostics_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from a Stable-Baselines3 zip.")
    parser.add_argument("model_zip", type=Path, nargs="?", help="Path to an SB3 *.zip checkpoint")
    parser.add_argument(
        "--batch-dir",
        type=Path,
        help="Recursively redraw every completed run under an rl_batch directory.",
    )
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument("--csv", type=Path, help="Optional output CSV path")
    args = parser.parse_args()
    if (args.model_zip is None) == (args.batch_dir is None):
        parser.error("provide exactly one of model_zip or --batch-dir")
    if args.batch_dir is not None and (args.output is not None or args.csv is not None):
        parser.error("--output and --csv are only valid for a single model_zip")
    return args


def find_completed_runs(batch_dir: Path) -> list[Path]:
    return sorted(
        model_zip
        for model_zip in batch_dir.rglob("*.zip")
        if (model_zip.with_suffix("") / "episode_log.csv").is_file()
    )


def main() -> None:
    args = parse_args()
    font_path = configure_plot_style()
    print(f"Font: {FONT_FAMILY} ({font_path})")

    if args.batch_dir is not None:
        model_zips = find_completed_runs(args.batch_dir)
        if not model_zips:
            raise RuntimeError(f"No completed run archives found under {args.batch_dir}")
        print(f"Completed runs: {len(model_zips)}")
        for index, model_zip in enumerate(model_zips, start=1):
            output = model_zip.with_name(f"{model_zip.stem}_training_curve.png")
            loss_output = plot_curve(model_zip, output, None)
            suffix = f" + {loss_output.name}" if loss_output is not None else ""
            print(f"[{index}/{len(model_zips)}] {output.name}{suffix}")
        return

    model_zip = args.model_zip
    assert model_zip is not None
    output = args.output or model_zip.with_name(f"{model_zip.stem}_training_curve.png")
    csv_output = args.csv
    loss_output = plot_curve(model_zip, output, csv_output)
    print(output)
    if loss_output is not None:
        print(loss_output)
    if csv_output is not None:
        print(csv_output)


if __name__ == "__main__":
    main()

# .\.venv\Scripts\python.exe scripts\plot_training_curve.py projects\cache\rl\ppo_20260702_223355.zip
