"""Validation of the instability index against the recorded rollouts.

Recomputes the index from stored margin traces under alternative definitions, computes
outcomes that do not use the XCoM, refits the two-segment description on each, relates
the index to those outcomes and to falls, bounds the effect of early termination, and
measures how predictions change along the training path.

    python -m analysis.gait120_index_validation --runs-dir <runs> --out <json>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import rankdata

from analysis.gait120_checkpoint_correlation import (
    WindowRow,
    _load_rows,
    _within_window_result,
    accuracy_level_analysis,
)

SEED = 20261105
BOOTSTRAP_DRAWS = 5000
FALL_LEAD_STEPS = 5


@dataclass(frozen=True)
class IndexDefinition:
    minimum: tuple[float, float] = (0.42, 0.40)
    end: tuple[float, float] = (0.30, 0.35)
    slope: tuple[float, float] = (0.15, 0.45)
    weights: tuple[float, float, float] = (0.55, 0.25, 0.20)
    tail: int = 10


def _scaled(factor: float) -> IndexDefinition:
    base = IndexDefinition()
    return IndexDefinition(
        minimum=(base.minimum[0] * factor, base.minimum[1] * factor),
        end=(base.end[0] * factor, base.end[1] * factor),
        slope=(base.slope[0] * factor, base.slope[1] * factor),
    )


DEFINITIONS = {
    "reported index": IndexDefinition(),
    "ramps 25% higher": _scaled(1.25),
    "ramps 25% lower": _scaled(0.75),
    "equal term weights": IndexDefinition(weights=(1 / 3, 1 / 3, 1 / 3)),
    "trailing 5 steps": IndexDefinition(tail=5),
    "trailing 20 steps": IndexDefinition(tail=20),
}


def index_trace(margin: np.ndarray, dt: float, d: IndexDefinition) -> np.ndarray:
    """Per-step instability index from a signed margin trace."""
    m = np.asarray(margin, dtype=np.float64)
    tail = max(3, min(d.tail, m.size))
    out = np.zeros(m.size, dtype=np.float64)
    for i in range(m.size):
        w = m[max(0, i - tail + 1) : i + 1]
        w = w[np.isfinite(w)]
        if w.size < 1:
            out[i] = 0.25
            continue
        slope = (w[-1] - w[0]) / ((w.size - 1) * dt) if w.size >= 2 else 0.0
        terms = (
            (-w.min() - d.minimum[0]) / d.minimum[1],
            (-w[-1] - d.end[0]) / d.end[1],
            (-slope - d.slope[0]) / d.slope[1],
        )
        terms = [min(1.0, max(0.0, t)) for t in terms]
        out[i] = min(1.0, max(0.0, sum(a * b for a, b in zip(d.weights, terms))))
    return out


def _integral(trace: np.ndarray, dt: float) -> float:
    values = np.asarray(trace, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 1:
        return float(values[0] * dt)
    return float(np.trapezoid(values, dx=dt))


def _rollout_outcomes(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        dt = float(np.asarray(z["dt"]).reshape(()))
        trace = {
            name: {
                "xcom": np.asarray(z[f"balance_xcom_margin_{suffix}_m"], dtype=np.float64),
                "com": np.asarray(z[f"balance_margin_{suffix}_m"], dtype=np.float64),
                "upright": np.asarray(z[f"upright_{suffix}"], dtype=np.float64),
                "root": np.asarray(z[f"root_z_{suffix}_m"], dtype=np.float64),
                "pose": np.asarray(z[f"termination_error_{suffix}"], dtype=np.float64),
                "stored": np.asarray(z[f"predicted_fall_risk_trace_{suffix}"], dtype=np.float64),
            }
            for name, suffix in (("reference", "ref"), ("override", "good"))
        }
        fall_reference = int(np.asarray(z["fall_step_ref"]).reshape(()))
        fall_override = int(np.asarray(z["fall_step_good"]).reshape(()))

    out: dict[str, float] = {}

    def excess(name: str, fn: Callable[[dict[str, np.ndarray]], np.ndarray]) -> None:
        a = _integral(fn(trace["override"]), dt)
        b = _integral(fn(trace["reference"]), dt)
        out[name] = a - b
        out[f"{name}|reference"] = b

    for name, definition in DEFINITIONS.items():
        excess(name, lambda t, d=definition: index_trace(t["xcom"], dt, d))
    excess("index on the CoM margin", lambda t: index_trace(t["com"], dt, IndexDefinition()))
    excess("negative XCoM margin", lambda t: np.maximum(0.0, -np.nan_to_num(t["xcom"])))
    excess("trunk tilt", lambda t: 1.0 - t["upright"])
    excess("root height drop", lambda t: np.maximum(0.0, t["root"][0] - t["root"]))
    excess("whole-body pose error", lambda t: t["pose"])

    steps = trace["override"]["xcom"].size
    head = trace["override"]["stored"][: max(1, steps - FALL_LEAD_STEPS)]
    return {
        "outcomes": out,
        "dt": dt,
        "recorded_steps": steps,
        "fall_reference": fall_reference,
        "fall_override": fall_override,
        "index_before_end": float(np.mean(head)),
        "stored_excess": _integral(trace["override"]["stored"], dt)
        - _integral(trace["reference"]["stored"], dt),
        "reference_index_rate": float(np.mean(trace["reference"]["stored"])),
    }


def _load_outcomes(panel: Path) -> dict[tuple[str, str], dict[str, Any]]:
    table = {}
    for path in sorted(panel.glob("stages/*/evals/*/compare.npz")):
        table[(path.parts[-4], path.parts[-2])] = _rollout_outcomes(path)
    return table


def _fit(rows: list[WindowRow], label: str) -> dict[str, Any]:
    result = accuracy_level_analysis(rows, np.random.default_rng(SEED))
    within = _within_window_result(rows)
    below = result["below_breakpoint"]
    above = result["above_breakpoint"]
    return {
        "label": label,
        "split_rmse_deg": result["breakpoint_rmse_deg"],
        "split_95pct_ci": result["breakpoint_95pct_ci"],
        "slope_below": below["slope_per_degree"],
        "slope_below_95pct_ci": below["slope_95pct_ci"],
        "slope_above": above["slope_per_degree"],
        "slope_above_95pct_ci": above["slope_95pct_ci"],
        "reversal_with_both_intervals_excluding_zero": bool(
            below["slope_95pct_ci"][1] < 0.0 < above["slope_95pct_ci"][0]
        ),
        "within_window_rho": within["mean_spearman_rho_fisher_back_transformed"],
        "within_window_p": within["p_value_two_sided"],
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _clustered(
    stat: Callable[[np.ndarray], float], clusters: np.ndarray
) -> tuple[float, list[float]]:
    rng = np.random.default_rng(SEED)
    members = {c: np.flatnonzero(clusters == c) for c in np.unique(clusters)}
    names = list(members)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = rng.integers(0, len(names), len(names))
        draws.append(stat(np.concatenate([members[names[i]] for i in chosen])))
    draws = np.asarray(draws, dtype=np.float64)
    draws = draws[np.isfinite(draws)]
    return stat(np.arange(clusters.size)), [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def _roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([positive, negative]))
    return float(
        (ranks[: positive.size].sum() - positive.size * (positive.size + 1) / 2)
        / (positive.size * negative.size)
    )


def _convergence(rows: list[WindowRow], table: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    keys = [(r.checkpoint, r.query_id) for r in rows]
    clusters = np.asarray([r.subject for r in rows])
    index = np.asarray([table[k]["outcomes"]["reported index"] for k in keys])
    out: dict[str, Any] = {}
    for name in ("trunk tilt", "root height drop", "whole-body pose error"):
        other = np.asarray([table[k]["outcomes"][name] for k in keys])
        value, ci = _clustered(lambda i, o=other: _spearman(index[i], o[i]), clusters)
        out[name] = {"spearman_rho": value, "ci_95pct": ci}
    return out


def _falls(rows: list[WindowRow], table: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    keys = [(r.checkpoint, r.query_id) for r in rows]
    clusters = np.asarray([r.subject for r in rows])
    ended = np.asarray([table[k]["recorded_steps"] < r.expected_steps for k, r in zip(keys, rows)])
    index = np.asarray([table[k]["index_before_end"] for k in keys])
    value, ci = _clustered(lambda i: _roc_auc(index[i][ended[i]], index[i][~ended[i]]), clusters)
    return {
        "rollouts_ended_early": int(ended.sum()),
        "windows_with_an_early_ending": len({k[1] for k, e in zip(keys, ended) if e}),
        "override_fell": int(sum(table[k]["fall_override"] >= 0 for k, e in zip(keys, ended) if e)),
        "reference_fell": int(sum(table[k]["fall_reference"] >= 0 for k, e in zip(keys, ended) if e)),
        "lead_steps_excluded": FALL_LEAD_STEPS,
        "mean_index_ended_early": float(index[ended].mean()),
        "mean_index_completed": float(index[~ended].mean()),
        "roc_auc": value,
        "roc_auc_ci_95pct": ci,
    }


def _early_termination_bound(
    rows: list[WindowRow], table: dict[tuple[str, str], dict[str, Any]]
) -> list[WindowRow]:
    """Excess instability with the unobserved override steps scored at the maximum index."""
    bounded = []
    for r in rows:
        t = table[(r.checkpoint, r.query_id)]
        missing = max(0.0, r.expected_steps - t["recorded_steps"])
        added = missing * t["dt"] * (1.0 - t["reference_index_rate"])
        bounded.append(replace(r, excess_instability_auc=r.excess_instability_auc + added))
    return bounded


def _path_properties(panel: Path) -> list[dict[str, Any]]:
    per_checkpoint: dict[str, list[list[float]]] = {}
    labels: list[str] = []
    for path in sorted((panel / "queries").glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            labels = [str(x) for x in z["checkpoint_labels"].tolist()]
            measured = np.asarray(z["knee_flexion_deg"], dtype=np.float64)
            fused = np.asarray(z["fused_prediction_deg"], dtype=np.float64)
        m = measured - measured.mean()
        for label, prediction in zip(labels, fused):
            p = prediction - prediction.mean()
            lags = np.arange(-30, 31)
            joint = [
                np.dot(m[max(0, -k) : m.size - max(0, k)], p[max(0, k) : p.size - max(0, -k)])
                / (m.size - abs(k))
                for k in lags
            ]
            per_checkpoint.setdefault(label, []).append(
                [
                    float(np.ptp(prediction) / np.ptp(measured)),
                    float(np.std(np.diff(prediction)) / np.std(np.diff(measured))),
                    float(lags[int(np.argmax(joint))] * 10.0),
                    float(np.mean(prediction - measured)),
                ]
            )
    return [
        {
            "checkpoint": label,
            "range_ratio": float(np.mean([v[0] for v in per_checkpoint[label]])),
            "frame_change_ratio": float(np.mean([v[1] for v in per_checkpoint[label]])),
            "best_lag_ms": float(np.mean([v[2] for v in per_checkpoint[label]])),
            "bias_deg": float(np.mean([v[3] for v in per_checkpoint[label]])),
        }
        for label in labels
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    panel = args.runs_dir.resolve() / "panel"

    rows = _load_rows(panel)
    table = _load_outcomes(panel)
    keys = [(r.checkpoint, r.query_id) for r in rows]

    identity = max(
        abs(table[k]["outcomes"]["reported index"] - table[k]["stored_excess"]) for k in keys
    )
    reference_by_window = {
        k[1]: table[k]["outcomes"]["reported index|reference"] for k in keys
    }
    reference = np.asarray(list(reference_by_window.values()))
    index_excess = np.asarray([r.excess_instability_auc for r in rows])

    names = list(DEFINITIONS) + [
        "index on the CoM margin",
        "negative XCoM margin",
        "trunk tilt",
        "root height drop",
        "whole-body pose error",
    ]
    fits = [
        _fit(
            [
                replace(r, excess_instability_auc=table[(r.checkpoint, r.query_id)]["outcomes"][name])
                for r in rows
            ],
            name,
        )
        for name in names
    ]

    report = {
        "largest_difference_from_stored_excess_s": float(identity),
        "reference_index_integral_s": {
            "mean": float(reference.mean()),
            "median": float(np.median(reference)),
            "min": float(reference.min()),
            "max": float(reference.max()),
            "share_above_0.1": float(np.mean(reference > 0.1)),
        },
        "excess_sd_across_rollouts_s": float(index_excess.std(ddof=1)),
        "definition_fits": fits,
        "convergence_with_outcomes_not_using_xcom": _convergence(rows, table),
        "falls": _falls(rows, table),
        "early_termination_upper_bound": _fit(
            _early_termination_bound(rows, table), "unobserved override steps at the maximum index"
        ),
        "path_properties": _path_properties(panel),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
