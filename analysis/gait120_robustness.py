"""Robustness of the reported association to motion type, tuning, and gains.

Refits the two-segment description on subsets of the fixed panel, resamples
matched clips instead of participants, summarizes the zero-error replay and the
tracking of commanded trajectories, refits at alternate controller gains, and
reports the sEMG improvement across participants and by time from peak knee
flexion. Applies the gain-selection rule to windows from the development cohort, and
refits on the second window panel and on the recurrent model.

    python -m analysis.gait120_robustness --runs-dir <runs> --out <json>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from analysis.gait120_checkpoint_correlation import (
    WindowRow,
    _load_rows,
    _within_window_result,
    accuracy_level_analysis,
    level_means,
    two_segment,
)

SEED = 20261104
LABELS = Path(__file__).with_name("cmu_clip_labels.json")
PHASE_BIN_FRAMES = 20
PHASE_MIN_ROWS = 5
PHASE_MIN_PARTICIPANTS = 30


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _refit(rows: list[WindowRow], label: str) -> dict[str, Any]:
    result = accuracy_level_analysis(rows, np.random.default_rng(SEED))
    within = _within_window_result(rows)
    return {
        "label": label,
        "n_windows": len({r.query_id for r in rows}),
        "n_participants": len({r.subject for r in rows}),
        "split_rmse_deg": result["breakpoint_rmse_deg"],
        "split_95pct_ci": result["breakpoint_95pct_ci"],
        "slope_above": result["above_breakpoint"]["slope_per_degree"],
        "slope_above_95pct_ci": result["above_breakpoint"]["slope_95pct_ci"],
        "slope_below": result["below_breakpoint"]["slope_per_degree"],
        "slope_below_95pct_ci": result["below_breakpoint"]["slope_95pct_ci"],
        "within_window_rho": within["mean_spearman_rho_fisher_back_transformed"],
        "within_window_95pct_ci": within["ci_95pct_rho"],
        "within_window_p": within["p_value_two_sided"],
        "mean_rmse_deg": result["mean_rmse_deg"],
        "mean_excess_instability": result["mean_excess_instability"],
    }


def _participant_ci(values: dict[str, list[float]], draws: int = 10_000) -> dict[str, float]:
    means = np.array([np.mean(v) for _, v in sorted(values.items())])
    rng = np.random.default_rng(SEED)
    drawn = means[rng.integers(0, means.size, size=(draws, means.size))].mean(axis=1)
    return {
        "mean": float(means.mean()),
        "ci_low": float(np.quantile(drawn, 0.025)),
        "ci_high": float(np.quantile(drawn, 0.975)),
    }


def _zero_error(run_dir: Path, max_tracking_deg: float) -> dict[str, Any]:
    rows = [_read(p) for p in sorted((run_dir / "oracle" / "evals").glob("*/summary.json"))]
    by_subject: dict[str, list[float]] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(
            row["simulation"]["excess_instability_auc"]["fused"]
        )
    excess = [row["simulation"]["excess_instability_auc"]["fused"] for row in rows]
    tracking = [row["simulation"]["fused"]["tracking_rmse_deg"] for row in rows]
    return {
        "gains": _read(run_dir / "oracle_protocol.json")["gains"],
        "n_windows": len(rows),
        "excess_participant_mean": _participant_ci(by_subject),
        "excess_window_sd": float(np.std(excess, ddof=1)),
        "excess_window_min": float(min(excess)),
        "excess_window_max": float(max(excess)),
        "tracking_mean_deg": float(np.mean(tracking)),
        "tracking_max_deg": float(max(tracking)),
        "windows_over_tracking_limit": int(sum(t > max_tracking_deg for t in tracking)),
        "reference_lost_balance": sorted(
            row["query_id"] for row in rows
            if row["simulation"]["reference"]["balance_loss_step"] != -1
        ),
        "override_ended_early": sorted(
            row["query_id"] for row in rows
            if row["simulation"]["fused"]["recorded_steps"] < row["match"]["length"]
        ),
        "over_tracking_limit": sorted(
            row["query_id"] for row in rows
            if row["simulation"]["fused"]["tracking_rmse_deg"] > max_tracking_deg
        ),
    }


def _command_tracking(panel_dir: Path, max_tracking_deg: float) -> list[dict[str, Any]]:
    by_checkpoint: dict[str, list[float]] = {}
    for path in panel_dir.glob("stages/*/evals/*/summary.json"):
        summary = _read(path)
        by_checkpoint.setdefault(summary["checkpoint"], []).append(
            summary["simulation"]["fused"]["tracking_rmse_deg"]
        )
    return [
        {
            "checkpoint": label,
            "mean_tracking_deg": float(np.mean(values)),
            "windows_over_limit": int(sum(v > max_tracking_deg for v in values)),
        }
        for label, values in sorted(by_checkpoint.items())
    ]


def _semg_distribution(runs: Path) -> dict[str, Any]:
    summary = _read(runs / "confirmation" / "ablation_summary.json")
    improvement = np.sort(np.asarray(summary["improvement_deg"], dtype=np.float64))[::-1]
    positive = improvement[improvement > 0]
    return {
        "mean_improvement_deg": float(improvement.mean()),
        "share_of_no_semg_rmse": float(
            improvement.mean() / summary["no_emg_mean_participant_rmse_deg"]
        ),
        "participants": int(improvement.size),
        "participants_improved": int(positive.size),
        "participants_worse": int(np.sum(improvement < 0)),
        "top_decile_share_of_net_gain": float(
            improvement[: improvement.size // 10].sum() / improvement.sum()
        ),
        "quartiles_deg": [float(v) for v in np.quantile(improvement, [0.25, 0.5, 0.75])],
    }


def _semg_by_gait_phase(runs: Path, checkpoint: str) -> dict[str, Any]:
    stored = np.load(runs / "training_path" / "checkpoints" / checkpoint / "test_predictions.npz")
    target = stored["target_deg"].astype(np.float64)
    fused = stored["fused_prediction_deg"].astype(np.float64)
    without = stored["no_emg_prediction_deg"].astype(np.float64)
    subjects = stored["subject_number"]
    frames = stored["target_frame"]

    reductions: dict[int, dict[int, float]] = {}
    for subject in np.unique(subjects):
        keep = np.flatnonzero(subjects == subject)
        offset = frames[keep] - frames[keep][np.argmax(target[keep])]
        bins = np.floor(offset / PHASE_BIN_FRAMES).astype(int)
        for bin_index in np.unique(bins):
            rows = keep[bins == bin_index]
            if rows.size < PHASE_MIN_ROWS:
                continue
            reductions.setdefault(int(bin_index), {})[int(subject)] = float(
                np.sqrt(np.mean((without[rows] - target[rows]) ** 2))
                - np.sqrt(np.mean((fused[rows] - target[rows]) ** 2))
            )

    table = []
    rng = np.random.default_rng(SEED)
    for bin_index, by_subject in sorted(reductions.items()):
        if len(by_subject) < PHASE_MIN_PARTICIPANTS:
            continue
        values = np.asarray(list(by_subject.values()), dtype=np.float64)
        drawn = values[rng.integers(0, values.size, size=(10_000, values.size))].mean(axis=1)
        table.append(
            {
                "start_ms_from_peak_flexion": int(bin_index * PHASE_BIN_FRAMES * 10),
                "n_participants": int(values.size),
                "mean_reduction_deg": float(values.mean()),
                "ci_low": float(np.quantile(drawn, 0.025)),
                "ci_high": float(np.quantile(drawn, 0.975)),
            }
        )
    return {"bin_ms": PHASE_BIN_FRAMES * 10, "bins": table}


def _independent_tuning(run_dir: Path, grid: list[dict[str, Any]], limit: float, labels: dict) -> dict[str, Any]:
    selection = _read(run_dir / "tuning_windows.json")
    table = []
    for gains in grid:
        rows = [
            _read(p)
            for p in (run_dir / f"kp{gains['kp']:g}_kd{gains['kd']:g}_force{gains['force']:g}")
            .glob("oracle/evals/*/summary.json")
        ]
        tracking = [row["simulation"]["fused"]["tracking_rmse_deg"] for row in rows]
        table.append(
            {
                "kp": gains["kp"],
                "kd": gains["kd"],
                "force": gains["force"],
                "windows": len(rows),
                "mean_tracking_deg": float(np.mean(tracking)),
                "max_tracking_deg": float(max(tracking)),
                "windows_over_limit": int(sum(t > limit for t in tracking)),
                "mean_excess_instability": float(
                    np.mean([row["simulation"]["excess_instability_auc"]["fused"] for row in rows])
                ),
            }
        )
    meeting = [row for row in table if row["windows_over_limit"] == 0]
    return {
        "participants": len({w["subject"] for w in selection["windows"]}),
        "clips_shared_with_the_panel": 0,
        "clip_categories": _categories([w["clip_id"] for w in selection["windows"]], labels),
        "table": table,
        "selected": min(meeting, key=lambda row: (row["kp"], row["kd"], row["force"])),
    }


def _categories(clips: list[str], labels: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for clip in clips:
        category = labels[clip]["category"]
        counts[category] = counts.get(category, 0) + 1
    return counts


def _paired_physics(rows: list[WindowRow]) -> dict[str, Any]:
    by_subject: dict[str, list[float]] = {}
    for row in rows:
        if row.has_no_emg:
            by_subject.setdefault(row.subject, []).append(
                row.no_emg_excess_auc - row.excess_instability_auc
            )
    values = np.array([np.mean(v) for _, v in sorted(by_subject.items())])
    test = stats.ttest_1samp(values, 0.0)
    half = stats.t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size)
    return {
        "mean": float(values.mean()),
        "ci_low": float(values.mean() - half),
        "ci_high": float(values.mean() + half),
        "t": float(test.statistic),
        "df": int(values.size - 1),
        "p": float(test.pvalue),
    }


def _second_panel(run_dir: Path, limit: float, labels: dict) -> dict[str, Any]:
    matching = _read(run_dir / "matching_preflight" / "summary.json")
    clip_of = {w["query_id"]: w["clip_id"] for w in matching["windows"]}
    rows = _load_rows_outside_oracle(run_dir)
    non_walking = {q for q, clip in clip_of.items() if labels[clip]["category"] != "walking"}
    panel = _read(run_dir / "panel_manifest.json")["protocol"]["selection"]
    zero = _zero_error(run_dir, limit)
    return {
        "trial": panel["trial"],
        "eligible_participants": panel["eligible_participants"],
        "participants": len({r.subject for r in rows}),
        "distinct_clips": len(set(clip_of.values())),
        "distinct_snippets": len({w["snippet_id"] for w in matching["windows"]}),
        "largest_snippet_share": max(
            sum(w["snippet_id"] == s for w in matching["windows"])
            for s in {w["snippet_id"] for w in matching["windows"]}
        ),
        "clip_categories_by_window": _categories(list(clip_of.values()), labels),
        "mean_match_knee_rmse_deg": matching["mean_knee_rmse_deg"],
        "rollouts_ended_early": int(sum(r.recorded_steps < r.expected_steps for r in rows)),
        "fit": _refit(rows, "second panel"),
        "fit_walking_clips": _refit(
            [r for r in rows if r.query_id not in non_walking], "second panel, walking clips"
        ),
        "paired_semg_physics": _paired_physics(rows),
        "zero_error": zero,
    }


NETWORK_MODELS = ("gru", "lstm", "mlp", "cnn", "transformer")
POOLED_DRAWS = 2000


def _pooled_decline(models: dict[str, list[WindowRow]]) -> dict[str, Any]:
    """Mean slope below the split across models, with one participant bootstrap for all."""
    prepared = {}
    for name, rows in models.items():
        by_checkpoint: dict[str, list[WindowRow]] = {}
        for row in rows:
            by_checkpoint.setdefault(row.checkpoint, []).append(row)
        labels = sorted(by_checkpoint)
        prepared[name] = (by_checkpoint, labels)
    first_by_checkpoint, first_labels = next(iter(prepared.values()))
    windows_by_subject: dict[str, list[str]] = {}
    for row in first_by_checkpoint[first_labels[0]]:
        windows_by_subject.setdefault(row.subject, []).append(row.query_id)
    subjects = sorted(windows_by_subject)

    def slopes(query_ids: list[str] | None) -> dict[str, float]:
        out = {}
        for name, (by_checkpoint, labels) in prepared.items():
            if query_ids is None:
                sample = by_checkpoint
            else:
                index = {label: {r.query_id: r for r in by_checkpoint[label]} for label in labels}
                sample = {label: [index[label][q] for q in query_ids] for label in labels}
            out[name] = two_segment(*level_means(sample, labels))[1]
        return out

    point = slopes(None)
    rng = np.random.default_rng(SEED)
    means = []
    for _ in range(POOLED_DRAWS):
        drawn = [subjects[i] for i in rng.integers(0, len(subjects), size=len(subjects))]
        means.append(float(np.mean(list(slopes([q for s in drawn for q in windows_by_subject[s]]).values()))))
    return {
        "models": list(models),
        "slope_below_by_model": point,
        "mean_slope_below": float(np.mean(list(point.values()))),
        "mean_slope_below_95pct_ci": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "models_with_negative_slope_below": int(sum(v < 0 for v in point.values())),
        "draws": POOLED_DRAWS,
    }


def _load_rows_outside_oracle(run_dir: Path) -> list[WindowRow]:
    return [r for r in _load_rows(run_dir) if r.checkpoint]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = args.runs_dir.resolve()

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    panel = runs / "panel"
    protocol = _read(panel / "physics_protocol.moving_target_pd_v2.json")
    limit = float(protocol["oracle_preflight"]["maximum_tracking_rmse_deg"])
    primary = protocol["checkpoints"][-1]

    rows = _load_rows(panel)
    clip_of = {
        p.parts[-2]: _read(p)["match"]["clip_id"]
        for p in (panel / "stages" / primary / "evals").glob("*/summary.json")
    }
    category = {query: labels[clip]["category"] for query, clip in clip_of.items()}
    tuning = {
        _read(p)["query_id"] for p in (panel / "oracle_preflight" / "evals").glob("*/summary.json")
    }
    zero_error = _zero_error(runs / "revision_kp400", limit)
    reference_lost = set(zero_error["reference_lost_balance"])
    ended_early = set(zero_error["override_ended_early"])
    over_limit = set(zero_error["over_tracking_limit"])

    def subset(excluded: set[str]) -> list[WindowRow]:
        return [r for r in rows if r.query_id not in excluded]

    walking = {q for q, c in category.items() if c == "walking"}
    non_walking = set(clip_of) - walking
    basketball = {q for q, c in category.items() if c == "basketball"}

    counts: dict[str, int] = {}
    for value in category.values():
        counts[value] = counts.get(value, 0) + 1

    fits = [
        _refit(rows, "all windows"),
        _refit([replace(r, subject=clip_of[r.query_id]) for r in rows], "all windows, resampling clips"),
        _refit(subset(non_walking), "walking clips"),
        _refit(subset(basketball), "excluding basketball clips"),
        _refit(subset(tuning), "excluding gain-tuning windows"),
        _refit(subset(non_walking | tuning), "walking clips, excluding gain-tuning windows"),
        _refit(subset(reference_lost), "excluding windows where the reference lost balance"),
        _refit(subset(reference_lost | ended_early), "also excluding the early-ending zero-error window"),
        _refit(subset(reference_lost | ended_early | over_limit), "also excluding zero-error tracking above the limit"),
    ]

    alternate_dir = runs / "revision_kp800"
    alternate_rows = _load_rows(alternate_dir)

    differing = set(_read(runs / "revision_gru" / "matching_rerun_differences.json")["windows"])
    report = {
        "independent_tuning": _independent_tuning(
            runs / "revision_tuning", _read(runs / "gain_sweep.json"), limit, labels
        ),
        "second_panel": _second_panel(runs / "revision_trial4", limit, labels),
        "network_models": {
            cell: {
                **_refit(_load_rows_outside_oracle(runs / f"revision_{cell}"), f"{cell} model"),
                "terminal_test_rmse_deg": _read(
                    runs / f"revision_{cell}_path" / "checkpoints" / "manifest.json"
                )["checkpoints"][-1]["fused_mean_participant_rmse_deg"],
                "checkpoints": len(
                    _read(runs / f"revision_{cell}_path" / "checkpoints" / "manifest.json")["labels"]
                ),
                "protocol": _read(runs / f"revision_{cell}_path" / "protocol.json"),
            }
            for cell in NETWORK_MODELS
        },
        "pooled_decline": _pooled_decline(
            {
                "ridge": rows,
                **{cell: _load_rows_outside_oracle(runs / f"revision_{cell}") for cell in NETWORK_MODELS},
            }
        ),
        "matching_rerun": {
            "windows_differing": len(differing),
            "fit_excluding_them": _refit(subset(differing), "excluding the two windows"),
        },
        "clip_categories_by_window": counts,
        "distinct_clips": len(set(clip_of.values())),
        "tuning_overlap": {
            "tuning_windows": len(tuning),
            "remaining_windows": len(set(clip_of) - tuning),
            "sharing_a_clip": sum(
                clip_of[q] in {clip_of[t] for t in tuning} for q in set(clip_of) - tuning
            ),
            "sharing_a_participant": sum(
                q.split("_")[0] in {t.split("_")[0] for t in tuning} for q in set(clip_of) - tuning
            ),
        },
        "largest_clip_share": max(
            sum(1 for c in clip_of.values() if c == clip) for clip in set(clip_of.values())
        ),
        "fits": fits,
        "zero_error_reported_gains": zero_error,
        "zero_error_alternate_gains": _zero_error(alternate_dir, limit),
        "command_tracking_by_checkpoint": _command_tracking(panel, limit),
        "alternate_gains": {
            "gains": _read(alternate_dir / "panel_protocol.json")["gains"],
            **_refit(alternate_rows, "alternate gains"),
            "command_tracking_by_checkpoint": _command_tracking(alternate_dir, limit),
        },
        "semg_distribution": _semg_distribution(runs),
        "semg_by_gait_phase": _semg_by_gait_phase(runs, primary),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
