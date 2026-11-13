"""Shared step-level and one-row summary metrics for RL inference."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np


ACTION_NAMES = {
    0: "ADVANCE",
    1: "RETRACT",
    2: "ROTATE_CW",
    3: "ROTATE_CCW",
    4: "NOOP",
}


def load_checkpoint_training_config(checkpoint: Path) -> dict:
    """Load metadata for both final checkpoints and nested ``best_model.zip``."""
    checkpoint = Path(checkpoint)
    candidates = (
        checkpoint.with_suffix("") / "training_config.json",
        checkpoint.parent / "training_config.json",
    )
    for config_path in candidates:
        if not config_path.is_file():
            continue
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def finite_or_blank(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def safety_margin_from_stats(stats: Mapping, guidewire) -> float:
    """Return clearance beyond wire radius + collision margin in millimetres."""
    min_sdf = finite_or_blank(stats.get("min_sdf"))
    if min_sdf == "":
        return float("nan")
    radius = float(getattr(guidewire, "radius", 0.0))
    margin = float(getattr(getattr(guidewire, "config", None), "collision_margin", 0.0))
    return -radius - margin - float(min_sdf)


def make_step_row(
    *,
    step: int,
    action_index: int,
    raw_action,
    action_applied: bool,
    requested_push_mm: float,
    speed_profile: str,
    speed_phase: str,
    stats: Mapping,
    guidewire,
    max_progress: float,
    reward: Optional[float] = None,
    total_reward: Optional[float] = None,
    metadata: Optional[Mapping] = None,
) -> dict:
    progress = float(stats.get("progress", 0.0))
    row = dict(metadata or {})
    row.update({
        "step": int(step),
        "action_index": int(action_index),
        "action_name": ACTION_NAMES.get(int(action_index), f"UNKNOWN_{action_index}"),
        "raw_action": str(raw_action),
        "action_applied": int(bool(action_applied)),
        "speed_profile": str(speed_profile),
        "speed_phase": str(speed_phase),
        "requested_push_mm": float(requested_push_mm),
        "reward": "" if reward is None else float(reward),
        "total_reward": "" if total_reward is None else float(total_reward),
        "progress": progress,
        "max_progress": float(max_progress),
        "success": int(bool(stats.get("success", False))),
        "tip_centerline_s_mm": finite_or_blank(stats.get("tip_centerline_s")),
        "distance_to_end_mm": finite_or_blank(stats.get("distance_to_end_by_tip")),
        "status": str(stats.get("status", "")),
        "blocked_this_step": int(int(action_index) == 0 and not bool(action_applied)),
        "angle_to_centerline_deg": finite_or_blank(stats.get("angle_to_centerline")),
        "tip_to_centerline_distance_mm": finite_or_blank(
            stats.get("tip_to_centerline_distance")
        ),
        "min_sdf_mm": finite_or_blank(stats.get("min_sdf")),
        "safety_margin_mm": finite_or_blank(safety_margin_from_stats(stats, guidewire)),
        "wall_slide_attempted": int(bool(stats.get("wall_slide_attempted", False))),
        "wall_slide_success": int(bool(stats.get("wall_slide_success", False))),
        "twist_angle_rad": finite_or_blank(stats.get("twist_angle")),
    })
    return row


def summarize_rollout(
    rows: Iterable[Mapping],
    *,
    success: bool,
    termination_reason: str,
    metadata: Optional[Mapping] = None,
    tick_ms: Optional[int] = None,
) -> dict:
    rows = list(rows)
    result = dict(metadata or {})
    progress = np.asarray([float(r.get("progress", 0.0)) for r in rows], dtype=float)
    actions = [int(r.get("action_index", -1)) for r in rows]
    counts = {idx: actions.count(idx) for idx in range(5)}
    advances = counts[0]
    rotations = counts[2] + counts[3]
    blocked = sum(int(r.get("blocked_this_step", 0)) for r in rows)
    wall_attempts = sum(int(r.get("wall_slide_attempted", 0)) for r in rows)
    wall_successes = sum(int(r.get("wall_slide_success", 0)) for r in rows)
    switches = sum(a != b for a, b in zip(actions, actions[1:]))
    safety = [
        float(v)
        for v in (r.get("safety_margin_mm", "") for r in rows)
        if v != "" and math.isfinite(float(v))
    ]
    distances = [
        float(v)
        for v in (r.get("distance_to_end_mm", "") for r in rows)
        if v != "" and math.isfinite(float(v))
    ]
    rewards = [
        float(v)
        for v in (r.get("reward", "") for r in rows)
        if v != "" and math.isfinite(float(v))
    ]
    if len(progress) > 1:
        trapezoid_sum = (
            0.5 * progress[0]
            + float(progress[1:-1].sum())
            + 0.5 * progress[-1]
        )
        progress_auc = float(trapezoid_sum / (len(progress) - 1))
    elif len(progress) == 1:
        progress_auc = float(progress[0])
    else:
        progress_auc = 0.0
    result.update({
        "success": int(bool(success)),
        "termination_reason": str(termination_reason),
        "steps": len(rows),
        "total_reward": float(sum(rewards)) if rewards else "",
        "final_progress": float(progress[-1]) if len(progress) else 0.0,
        "max_progress": float(progress.max()) if len(progress) else 0.0,
        "progress_auc_step_normalized": progress_auc,
        "estimated_execution_time_s": (
            "" if tick_ms is None else len(rows) * float(tick_ms) / 1000.0
        ),
        "advance_count": counts[0],
        "retract_count": counts[1],
        "rotate_cw_count": counts[2],
        "rotate_ccw_count": counts[3],
        "rotation_count": rotations,
        "rotation_rate": rotations / len(rows) if rows else 0.0,
        "action_switch_count": switches,
        "action_switch_rate": switches / max(len(rows) - 1, 1) if rows else 0.0,
        "proximal_advance_count": sum(
            int(r.get("action_index", -1) == 0 and r.get("speed_phase") == "proximal")
            for r in rows
        ),
        "distal_advance_count": sum(
            int(r.get("action_index", -1) == 0 and r.get("speed_phase") == "distal")
            for r in rows
        ),
        "requested_advance_distance_mm": sum(
            float(r.get("requested_push_mm", 0.0)) for r in rows
        ),
        "blocked_advance_count": blocked,
        "blocked_advance_rate": blocked / advances if advances else 0.0,
        "wall_slide_attempt_count": wall_attempts,
        "wall_slide_success_count": wall_successes,
        "wall_slide_success_rate": wall_successes / wall_attempts if wall_attempts else 0.0,
        "minimum_safety_margin_mm": min(safety) if safety else "",
        "mean_safety_margin_mm": float(np.mean(safety)) if safety else "",
        "final_distance_to_end_mm": distances[-1] if distances else "",
        "minimum_distance_to_end_mm": min(distances) if distances else "",
    })
    return result


def write_inference_csvs(
    rows: list[dict],
    summary: dict,
    rollout_path: Path,
    summary_path: Optional[Path] = None,
) -> tuple[Path, Path]:
    rollout_path = Path(rollout_path)
    summary_path = (
        Path(summary_path)
        if summary_path is not None
        else rollout_path.with_name(f"{rollout_path.stem}_summary.csv")
    )
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    step_fields = list(rows[0]) if rows else ["step"]
    with rollout_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=step_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return rollout_path, summary_path
