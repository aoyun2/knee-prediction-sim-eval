"""Zero-error replay, gain sensitivity, replication panels, and independent tuning.

``oracle`` commands each panel window's matched clip with its own recorded knee
trajectory. ``panel`` replays every checkpoint-window pair. ``match`` writes the motion
matches of a new panel. ``tuning-select`` draws ten windows from development-cohort
participants and matches them to clips outside the reported panel. ``tuning`` replays
those windows with their matched clips' own trajectories at one setting of the gains.

    python -m mocap_phys_eval.run_gait120_revision_checks oracle \
        --panel-dir <panel> --run-dir <out> --artifacts-dir <artifacts>
    python -m mocap_phys_eval.run_gait120_revision_checks panel \
        --panel-dir <panel> --run-dir <out> --artifacts-dir <artifacts> \
        --kp 800 --kd 40 --force 160
    python -m mocap_phys_eval.run_gait120_revision_checks tuning-select \
        --panel-dir <panel> --run-dir <out> --artifacts-dir <artifacts> --cache-dir <cache>
    python -m mocap_phys_eval.run_gait120_revision_checks tuning \
        --panel-dir <panel> --run-dir <out> --artifacts-dir <artifacts> --kp 400 --kd 20 --force 160
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

_STATE: dict[str, Any] = {}
TUNING_SEED = 20261105
TUNING_WINDOWS = 10
TUNING_WINDOW_FRAMES = 100
TUNING_WINDOW_HZ = 100
TUNING_TOP_K = 500
HISTORY_AND_HORIZON_FRAMES = 69


def _set_gains(kp: float, kd: float, force: float) -> None:
    os.environ["PROSTHETIC_KNEE_KP"] = str(kp)
    os.environ["PROSTHETIC_KNEE_KD"] = str(kd)
    os.environ["PROSTHETIC_KNEE_FORCE"] = str(force)


def _init_worker(panel_dir: str, artifacts_dir: str) -> None:
    from . import run_gait120_residual_fusion as runner
    from .config import EvalConfig
    from .run import _ensure_reference_bank
    from .utils import set_global_determinism

    runner.MOVING_TARGET_PD = True
    runner.VERSION = runner.MOVING_TARGET_PD_VERSION
    set_global_determinism(seed=0)
    _, queries = runner._load_panel(Path(panel_dir))
    matching = json.loads(
        (Path(panel_dir) / "matching_preflight" / "summary.json").read_text(encoding="utf-8")
    )
    cfg = EvalConfig(artifacts_dir=Path(artifacts_dir), device="cpu")
    bank = _ensure_reference_bank(cfg, out_root=Path(artifacts_dir))
    _STATE.update(
        runner=runner,
        queries=queries,
        matches={str(row["query_id"]): row for row in matching["windows"]},
        cfg=cfg,
        bank=bank,
        bank_index={str(bank.snippet_id[i]): i for i in range(len(bank))},
    )


def _init_tuning_worker(selection_path: str, artifacts_dir: str) -> None:
    from . import run_gait120_residual_fusion as runner
    from .config import EvalConfig
    from .run import _ensure_reference_bank
    from .utils import set_global_determinism

    runner.MOVING_TARGET_PD = True
    set_global_determinism(seed=0)
    windows = json.loads(Path(selection_path).read_text(encoding="utf-8"))["windows"]
    cfg = EvalConfig(artifacts_dir=Path(artifacts_dir), device="cpu")
    bank = _ensure_reference_bank(cfg, out_root=Path(artifacts_dir))
    _STATE.update(
        runner=runner,
        queries=[SimpleNamespace(query_id=w["query_id"], subject=w["subject"]) for w in windows],
        matches={str(w["query_id"]): w for w in windows},
        cfg=cfg,
        bank=bank,
        bank_index={str(bank.snippet_id[i]): i for i in range(len(bank))},
    )


def _select_tuning_windows(
    panel_dir: Path, run_dir: Path, artifacts_dir: Path, cache_dir: Path
) -> None:
    from . import run_gait120_residual_fusion as runner
    from .config import EvalConfig
    from .matching import motion_match_one_window
    from .run import _ensure_reference_bank
    from .utils import resample_linear

    panel_clips = {
        str(row["clip_id"])
        for row in json.loads(
            (panel_dir / "matching_preflight" / "summary.json").read_text(encoding="utf-8")
        )["windows"]
    }
    cfg = EvalConfig(artifacts_dir=artifacts_dir, device="cpu")
    bank = _ensure_reference_bank(cfg, out_root=artifacts_dir)
    match_hz = float(np.median(np.asarray(bank.sample_hz, dtype=np.float64)))
    rng = np.random.default_rng(TUNING_SEED)

    windows = []
    for number in rng.permutation(np.arange(1, 31)).tolist():
        subject = f"S{number:03d}"
        with np.load(cache_dir / f"{subject}.npz", allow_pickle=False) as stored:
            rows = np.flatnonzero(np.asarray(stored["trial_index"]) == 5)
            first = HISTORY_AND_HORIZON_FRAMES
            starts = list(range(first, rows.size - TUNING_WINDOW_FRAMES + 1, 10))
            if not starts:
                continue
            start = int(rng.choice(starts))
            chosen = rows[start : start + TUNING_WINDOW_FRAMES]
            knee = np.asarray(stored["knee_flexion_deg"], dtype=np.float32)[chosen]
            thigh = np.asarray(stored["thigh_pitch_deg"], dtype=np.float32)[chosen]
            hz = float(TUNING_WINDOW_HZ)
        knee_r = resample_linear(knee, src_hz=hz, dst_hz=match_hz)
        thigh_r = resample_linear(thigh, src_hz=hz, dst_hz=match_hz)
        length = int(min(knee_r.size, thigh_r.size))
        candidates = motion_match_one_window(
            bank=bank,
            query_thigh_deg=thigh_r[:length],
            query_thigh_quat_wxyz=None,
            query_knee_deg=knee_r[:length],
            top_k=TUNING_TOP_K,
            local_refine_radius=int(cfg.match_local_refine_radius),
            feature_mode=str(cfg.match_feature_mode),
            knee_weight=float(cfg.match_knee_weight),
            thigh_weight=float(cfg.match_thigh_weight),
        )
        outside = [c for c in candidates if str(c.clip_id) not in panel_clips]
        if not outside:
            continue
        query = SimpleNamespace(
            query_id=f"{subject}_trial05_start{start:04d}", panel_index=len(windows), subject=subject
        )
        windows.append(runner._candidate_row(query, outside[0], length))
        if len(windows) == TUNING_WINDOWS:
            break
    if len(windows) != TUNING_WINDOWS:
        raise RuntimeError(f"Selected {len(windows)} of {TUNING_WINDOWS} tuning windows")
    run_dir.mkdir(parents=True, exist_ok=True)
    runner._atomic_json(
        run_dir / "tuning_windows.json",
        {
            "seed": TUNING_SEED,
            "cohort": "development S001-S030, trial 5",
            "clips_excluded": sorted(panel_clips),
            "windows": windows,
        },
    )


def _panel_cell(checkpoint_index: int, query_index: int, run_dir: str) -> tuple[str, str]:
    runner = _STATE["runner"]
    query = _STATE["queries"][query_index]
    label = runner.CHECKPOINT_LABELS[checkpoint_index]
    runner._evaluate(
        query,
        checkpoint_index,
        label,
        _STATE["matches"][query.query_id],
        Path(run_dir) / "stages" / label / "evals" / query.query_id,
        _STATE["cfg"],
        _STATE["bank"],
        _STATE["bank_index"],
    )
    return label, query.query_id


def _oracle_cell(query_index: int, run_dir: str) -> tuple[str, str]:
    from .recording import record_compare_rollout
    from .sim import OverrideSpec, load_expert_policy

    runner = _STATE["runner"]
    cfg, bank = _STATE["cfg"], _STATE["bank"]
    query = _STATE["queries"][query_index]
    match = _STATE["matches"][query.query_id]
    q_dir = Path(run_dir) / "oracle" / "evals" / query.query_id
    summary_path = q_dir / "summary.json"
    if summary_path.exists():
        return "oracle", query.query_id
    q_dir.mkdir(parents=True, exist_ok=False)
    bi = _STATE["bank_index"][match["snippet_id"]]
    start = int(match["start_step_in_snippet"])
    length = int(match["length"])
    reference = np.asarray(bank.knee_deg[bi], dtype=np.float32)[start : start + length]
    policy = load_expert_policy(
        Path(str(np.asarray(bank.expert_model_path[bi]).reshape(()))), device="cpu"
    )
    recording = q_dir / "oracle_compare.npz"
    record_compare_rollout(
        out_npz_path=recording,
        clip_id=str(match["clip_id"]),
        start_step=int(np.asarray(bank.start_step[bi]).reshape(())) + start,
        end_step=int(np.asarray(bank.end_step[bi]).reshape(())),
        primary_steps=length,
        warmup_steps=max(0, start),
        policy=policy,
        override=OverrideSpec(str(cfg.knee_actuator), 1.0, 0.0),
        knee_good_query_deg=reference,
        knee_bad_query_deg=reference,
        width=int(cfg.render_width),
        height=int(cfg.render_height),
        camera_id=int(cfg.render_camera_id),
        deterministic_policy=True,
        seed=0,
        run_bad=False,
        panel_labels=("Reference", "Exact knee target", "Unused"),
        moving_target_pd=True,
    )
    runner._atomic_json(
        summary_path,
        {
            "query_id": query.query_id,
            "subject": query.subject,
            "match": match,
            "simulation": runner._record_metrics(recording, reference),
        },
    )
    return "oracle", query.query_id


def _remove_incomplete(root: Path) -> int:
    removed = 0
    for q_dir in root.glob("**/evals/*"):
        if q_dir.is_dir() and not (q_dir / "summary.json").exists():
            shutil.rmtree(q_dir)
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("oracle", "panel", "match", "tuning-select", "tuning"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--kp", type=float, default=400.0)
    parser.add_argument("--kd", type=float, default=20.0)
    parser.add_argument("--force", type=float, default=160.0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    _set_gains(args.kp, args.kd, args.force)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("MUJOCO_GL", "glfw")

    from . import run_gait120_residual_fusion as runner

    run_dir = args.run_dir.resolve()
    panel_dir = args.panel_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "tuning-select":
        _select_tuning_windows(panel_dir, run_dir, args.artifacts_dir.resolve(), args.cache_dir.resolve())
        return
    if args.mode == "tuning":
        selection = run_dir / "tuning_windows.json"
        gain_dir = run_dir / f"kp{args.kp:g}_kd{args.kd:g}_force{args.force:g}"
        count = len(json.loads(selection.read_text(encoding="utf-8"))["windows"])
        runner._atomic_json(
            gain_dir / "oracle_protocol.json",
            {"gains": {"kp": args.kp, "kd": args.kd, "force": args.force}},
        )
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_tuning_worker,
            initargs=(str(selection), str(args.artifacts_dir.resolve())),
        ) as pool:
            for future in as_completed(
                [pool.submit(_oracle_cell, i, str(gain_dir)) for i in range(count)]
            ):
                future.result()
        return
    if args.mode == "match":
        from .config import EvalConfig
        from .run import _ensure_reference_bank

        runner.MOVING_TARGET_PD = True
        _, match_queries = runner._load_panel(panel_dir)
        cfg = EvalConfig(artifacts_dir=args.artifacts_dir.resolve(), device="cpu")
        bank = _ensure_reference_bank(cfg, out_root=args.artifacts_dir.resolve())
        match_hz = float(np.median(np.asarray(bank.sample_hz, dtype=np.float64)))
        runner._matching_preflight(match_queries, panel_dir, cfg, bank, match_hz)
        return
    removed = _remove_incomplete(run_dir)
    _, queries = runner._load_panel(panel_dir)
    runner._atomic_json(
        run_dir / f"{args.mode}_protocol.json",
        {
            "mode": args.mode,
            "gains": {"kp": args.kp, "kd": args.kd, "force": args.force},
            "panel_dir": panel_dir.name,
            "moving_target_pd": True,
            "windows": len(queries),
            "checkpoints": list(runner.CHECKPOINT_LABELS),
            "code_sha256": {
                name: runner._sha256(Path(runner.__file__).parent / name)
                for name in (
                    "run_gait120_residual_fusion.py",
                    "run_gait120_revision_checks.py",
                    "recording.py",
                    "sim.py",
                    "run.py",
                    "matching.py",
                )
            },
        },
    )

    if args.mode == "oracle":
        jobs = [(_oracle_cell, (i, str(run_dir))) for i in range(len(queries))]
    else:
        jobs = [
            (_panel_cell, (c, i, str(run_dir)))
            for c in range(len(runner.CHECKPOINT_LABELS))
            for i in range(len(queries))
        ]

    started = time.time()
    print(f"{len(jobs)} rollouts, {args.workers} workers, {removed} incomplete removed", flush=True)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(panel_dir), str(args.artifacts_dir.resolve())),
    ) as pool:
        futures = [pool.submit(fn, *job_args) for fn, job_args in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            label, query_id = future.result()
            if done % 10 == 0 or done == len(jobs):
                hours = (time.time() - started) / 3600
                print(f"{done}/{len(jobs)}  {hours:.2f} h  {label} {query_id}", flush=True)


if __name__ == "__main__":
    main()
