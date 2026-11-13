"""Distance-based guidewire advance-step schedules for RL experiments.

The switch variable is normalized centerline arc-length progress, not the
Euclidean distance from the entry.  Once an episode reaches the switch point,
the distal (slow) phase remains active even if the guidewire later retracts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Union


@dataclass(frozen=True)
class SegmentedSpeedConfig:
    """Two-stage forward-step schedule used by training and inference."""

    profile: str
    proximal_push_mm: float
    distal_push_mm: float
    switch_progress: float

    def __post_init__(self):
        if self.proximal_push_mm <= 0.0:
            raise ValueError("proximal_push_mm must be > 0")
        if self.distal_push_mm <= 0.0:
            raise ValueError("distal_push_mm must be > 0")
        if not 0.0 <= self.switch_progress <= 1.0:
            raise ValueError("switch_progress must be in [0, 1]")

    def phase(self, best_progress: float, distal_latched: bool = False) -> str:
        return (
            "distal"
            if distal_latched or float(best_progress) >= self.switch_progress
            else "proximal"
        )

    def push_step_mm(self, best_progress: float, distal_latched: bool = False) -> float:
        return (
            self.distal_push_mm
            if self.phase(best_progress, distal_latched) == "distal"
            else self.proximal_push_mm
        )

    def to_dict(self) -> dict:
        return asdict(self)


# B0 is the original fixed-step baseline. The named profiles change only the requested
# ADVANCE distance; reward parameters, rotation angle, observations, and PPO
# hyperparameters stay unchanged.
SPEED_PROFILE_PRESETS: dict[str, SegmentedSpeedConfig] = {
    "B0": SegmentedSpeedConfig("B0", 0.50, 0.50, 0.80),
    "A1": SegmentedSpeedConfig("A1", 0.75, 0.50, 0.80),
    "A2": SegmentedSpeedConfig("A2", 1.00, 0.50, 0.80),
    "A3": SegmentedSpeedConfig("A3", 0.75, 0.25, 0.80),
    "A4": SegmentedSpeedConfig("A4", 1.50, 0.50, 0.80),
    "A5": SegmentedSpeedConfig("A5", 2.00, 0.50, 0.80),
    "A6": SegmentedSpeedConfig("A6", 2.25, 0.50, 0.80),
    "A7": SegmentedSpeedConfig("A7", 2.50, 0.50, 0.80),
    "P75": SegmentedSpeedConfig("P75", 2.00, 0.50, 0.75),
    "P85": SegmentedSpeedConfig("P85", 2.00, 0.50, 0.85),
    "P90": SegmentedSpeedConfig("P90", 2.00, 0.50, 0.90),
    "P95": SegmentedSpeedConfig("P95", 2.00, 0.50, 0.95),
}

SpeedConfigInput = Optional[
    Union[str, SegmentedSpeedConfig, Mapping[str, object]]
]


def resolve_speed_config(value: SpeedConfigInput = None) -> SegmentedSpeedConfig:
    """Resolve a preset name or a custom mapping into a validated config.

    Custom mappings accept ``proximal_push_mm``/``distal_push_mm`` and the
    friendlier aliases ``fast_push_mm``/``slow_push_mm``.
    """
    if value is None:
        return SPEED_PROFILE_PRESETS["B0"]
    if isinstance(value, SegmentedSpeedConfig):
        return value
    if isinstance(value, str):
        name = value.strip().upper()
        if name not in SPEED_PROFILE_PRESETS:
            raise ValueError(
                f"Unknown speed profile {value!r}; choose "
                f"{', '.join(SPEED_PROFILE_PRESETS)} or provide a custom mapping"
            )
        return SPEED_PROFILE_PRESETS[name]
    if not isinstance(value, Mapping):
        raise TypeError("speed_config must be None, a profile name, or a mapping")

    profile = str(value.get("profile", "CUSTOM")).strip().upper() or "CUSTOM"
    base = SPEED_PROFILE_PRESETS.get(profile, SPEED_PROFILE_PRESETS["B0"])
    proximal = value.get(
        "proximal_push_mm",
        value.get("fast_push_mm", base.proximal_push_mm),
    )
    distal = value.get(
        "distal_push_mm",
        value.get("slow_push_mm", base.distal_push_mm),
    )
    switch = value.get("switch_progress", base.switch_progress)

    # Any numerical override is an explicit custom condition.  Keeping this
    # label prevents a modified A1 from being mixed with the canonical A1 rows.
    canonical = (
        profile in SPEED_PROFILE_PRESETS
        and float(proximal) == base.proximal_push_mm
        and float(distal) == base.distal_push_mm
        and float(switch) == base.switch_progress
    )
    resolved_name = profile if canonical else "CUSTOM"
    return SegmentedSpeedConfig(
        profile=resolved_name,
        proximal_push_mm=float(proximal),
        distal_push_mm=float(distal),
        switch_progress=float(switch),
    )


def build_speed_config(
    profile: str = "B0",
    proximal_push_mm: Optional[float] = None,
    distal_push_mm: Optional[float] = None,
    switch_progress: Optional[float] = None,
) -> SegmentedSpeedConfig:
    """Build a preset or a user-overridden custom condition."""
    base = resolve_speed_config(profile if profile.upper() != "CUSTOM" else "B0")
    return resolve_speed_config({
        "profile": profile,
        "proximal_push_mm": (
            base.proximal_push_mm if proximal_push_mm is None else proximal_push_mm
        ),
        "distal_push_mm": (
            base.distal_push_mm if distal_push_mm is None else distal_push_mm
        ),
        "switch_progress": (
            base.switch_progress if switch_progress is None else switch_progress
        ),
    })
