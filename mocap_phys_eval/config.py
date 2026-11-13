from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_MODELS_DIR = Path(os.environ.get("MOCAPACT_MODELS_DIR", "mocapact_models")).expanduser()
_ARTIFACTS_DIR = Path(os.environ.get("MOCAP_PHYS_EVAL_ARTIFACTS_DIR", str(Path("artifacts") / "phys_eval_v2"))).expanduser()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _env_opt_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


@dataclass(frozen=True)
class EvalConfig:
    # Outputs.
    artifacts_dir: Path = _ARTIFACTS_DIR

    device: str = os.environ.get("MOCAP_PHYS_EVAL_DEVICE", "cpu")

    # MoCapAct per-snippet experts (~2589). The pipeline will auto-download+extract if missing.
    experts_root: Path = _MODELS_DIR / "experts"
    experts_downloads_dir: Path = _MODELS_DIR / "_downloads"

    # We evaluate a "prosthetic" right knee by overriding the knee DoF only.
    knee_actuator: str = "walker/rtibiarx"  # right knee flexion



    # Motion matching.
    match_top_k: int = 12
    match_local_refine_radius: int = 30
    # Scalar thigh pitch plus knee dynamics for candidate scoring, with the score driven by knee RMSE.
    match_feature_mode: str = "thigh_knee_d"
    match_knee_weight: float = _env_float("MOCAP_PHYS_EVAL_MATCH_KNEE_WEIGHT", 1.0)
    match_thigh_weight: float = _env_float("MOCAP_PHYS_EVAL_MATCH_THIGH_WEIGHT", 0.0)

    # Compare rollout recording.
    render_width: int = 480
    render_height: int = 360
    render_camera_id: int = 2

    # Note: when using per-snippet experts, warmup is determined by the chosen
    # window's offset within the matched snippet (we always start at snippet start).



