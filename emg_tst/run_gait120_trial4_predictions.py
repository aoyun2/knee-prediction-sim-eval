"""Export training-path predictions for trial 4, the held-out trial of the second panel.

Refits the residual-fusion training path on trials 1-3 and writes the checkpoint layout
``prepare_gait120_physics_panel`` reads, with trial 4 in place of trial 5.

    python -m emg_tst.run_gait120_trial4_predictions --cache-dir <cache> \
        --confirmation-run-dir <confirmation> --reference-path-dir <path> --run-dir <out>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import emg_tst.gait120_experiment as gait
import emg_tst.gait120_training_path as tpath
import emg_tst.run_gait120_residual_fusion as fusion
from emg_tst.run_gait120_training_path import _selected_hyperparameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--confirmation-run-dir", type=Path, required=True)
    parser.add_argument("--reference-path-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(
        (args.reference_path_dir / "checkpoints" / "manifest.json").read_text(encoding="utf-8")
    )
    protocol = json.loads((args.reference_path_dir / "protocol.json").read_text(encoding="utf-8"))
    kinematic_alpha, residual_alpha, gamma = _selected_hyperparameters(args.confirmation_run_dir)
    gait.INPUT_FRAMES = fusion.KINEMATIC_FRAMES
    subjects = list(range(31, 121))
    train, trial4, trial5, _ = gait._build_examples(args.cache_dir.resolve(), subjects)

    path = tpath.fit_training_path(
        train,
        kinematic_alpha=kinematic_alpha,
        residual_alpha=residual_alpha,
        max_steps=int(protocol["max_steps"]),
        checkpoint_mode=str(reference["checkpoint_mode"]),
        n_checkpoints=len(reference["labels"]),
    )

    worst = 0.0
    rows = []
    for order, recorded_index in enumerate(path.checkpoint_indices.tolist()):
        label = tpath.checkpoint_label(order, int(path.steps[recorded_index]))
        if label != reference["labels"][order]:
            raise RuntimeError(f"Checkpoint schedule changed at {label}")
        kinematic, residual, caps = tpath.materialize_checkpoint(path, order, train)
        _, _, repeat, _ = fusion._evaluate(
            trial5, kinematic=kinematic, residual=residual, gamma=gamma, correction_caps=caps
        )
        with np.load(args.reference_path_dir / "checkpoints" / label / "test_predictions.npz") as z:
            worst = max(worst, float(np.max(np.abs(repeat - z["fused_prediction_deg"]))))

        fused_metrics, no_emg_metrics, fused, no_emg = fusion._evaluate(
            trial4, kinematic=kinematic, residual=residual, gamma=gamma, correction_caps=caps
        )
        gait._atomic_npz(
            args.run_dir / "checkpoints" / label / "test_predictions.npz",
            target_deg=np.asarray(trial4.y_deg, dtype=np.float32),
            subject_number=np.asarray(trial4.subject_number, dtype=np.int16),
            trial_index=np.asarray(trial4.trial_index, dtype=np.int16),
            input_end_frame=np.asarray(trial4.input_end_frame, dtype=np.int32),
            target_frame=np.asarray(trial4.target_frame, dtype=np.int32),
            fused_prediction_deg=np.asarray(fused, dtype=np.float32),
            no_emg_prediction_deg=np.asarray(no_emg, dtype=np.float32),
        )
        rows.append(
            {
                "label": label,
                "fused_mean_participant_rmse_deg": fused_metrics["mean_participant_rmse_deg"],
                "no_emg_mean_participant_rmse_deg": no_emg_metrics["mean_participant_rmse_deg"],
            }
        )
        print(label, f"{rows[-1]['fused_mean_participant_rmse_deg']:.3f}", flush=True)

    gait._atomic_json(
        args.run_dir / "checkpoints" / "manifest.json",
        {
            "labels": reference["labels"],
            "primary_checkpoint": reference["primary_checkpoint"],
            "checkpoint_mode": reference["checkpoint_mode"],
            "checkpoints": rows,
        },
    )
    gait._atomic_json(
        args.run_dir / "protocol.json",
        {
            "held_out_trial": 4,
            "largest_difference_from_stored_trial5_predictions_deg": worst,
            "kinematic_alpha": kinematic_alpha,
            "residual_alpha": residual_alpha,
            "gamma": gamma,
        },
    )
    print(f"largest difference from stored trial 5 predictions: {worst:.3e} deg")


if __name__ == "__main__":
    main()
