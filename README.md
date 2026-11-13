# Gait120 residual-fusion simulation study

Analysis code for a paired study of 100-ms knee-angle prediction and whole-body
simulation. A residual-fusion model with hyperparameters fixed on 30 Gait120
participants was evaluated on later trials from a separate 90-participant cohort.
The predicted knee angle is the tracking target of a PD override on the right knee
of a simulated humanoid, against a paired unmodified reference on the same matched
motion. One fixed panel of 80 windows is replayed at 14 checkpoints along the
model's training path.

The experiment used healthy-participant data. It did not test a prosthetic device
or estimate clinical fall risk.

## Repository map

| Path | Purpose |
|---|---|
| `emg_tst/fetch_gait120.py` | Gait120 download from figshare, with checksums |
| `emg_tst/preprocess_gait120.py` | Prediction cache from the Gait120 release |
| `emg_tst/gait120_data.py` | Gait120 MAT-file loading and causal sEMG processing |
| `emg_tst/gait120_experiment.py` | Example construction, ridge fitting, and participant metrics |
| `emg_tst/run_gait120_residual_fusion.py` | Development and confirmation sEMG ablation |
| `emg_tst/run_gait120_temporal_control.py` | 500-ms timing control |
| `emg_tst/run_gait120_semg_controls.py` | sEMG surrogate negative controls |
| `emg_tst/run_gait120_kinematic_input_check.py` | Kinematic stage reading thigh kinematics instead of knee history |
| `emg_tst/gait120_training_path.py` | Gradient-descent path and checkpoint selection |
| `emg_tst/run_gait120_training_path.py` | Per-checkpoint models and test predictions |
| `emg_tst/prepare_gait120_physics_panel.py` | Fixed participant-balanced simulation panel |
| `emg_tst/run_gait120_trial4_predictions.py` | Training-path predictions for the second panel |
| `emg_tst/run_gait120_network_path.py` | Neural-network models and their training paths |
| `mocap_phys_eval/run_gait120_residual_fusion.py` | Motion matching and paired MuJoCo rollouts |
| `mocap_phys_eval/run_gait120_revision_checks.py` | Zero-error replay, alternate gains, replication panels, independent tuning |
| `analysis/gait120_checkpoint_correlation.py` | Association along the training path |
| `analysis/gait120_conventional_paired_statistics.py` | Participant-level inferential summaries |
| `analysis/gait120_sensitivity.py` | Early-termination and weighting sensitivity |
| `analysis/gait120_robustness.py` | Motion-type, tuning, tracking, gain, panel, and model robustness |
| `analysis/gait120_index_validation.py` | Instability index against alternative outcomes and falls |
| `analysis/cmu_clip_labels.json` | CMU descriptions of the matched clips |
| `analysis/gait120_figures.py` | Manuscript figures |
| `analysis/make_provenance.py` | Record of code and environment |
| `docs/EXPERIMENT_PROTOCOL.md` | Analysis protocol |
| `tests/` | Unit tests |

## Data

- [Gait120 dataset](https://doi.org/10.6084/m9.figshare.27677016)
- [Gait120 data descriptor](https://doi.org/10.1038/s41597-025-05391-0)
- [MoCapAct](https://microsoft.github.io/MoCapAct/)

Predictor inputs and targets are not interpolated or time-normalized. Three rate
conversions occur elsewhere: the 2000-Hz sEMG envelope contributes its final sample
in each 20-sample frame block, 100-Hz Gait120 queries are matched against the 200-Hz
MoCapAct reference grid, and 100-Hz commands are resampled onto the 0.03-s simulation
grid.

## Environments

```bash
# Prediction and statistics (Python 3.12)
python -m pip install -r requirements.txt

# Figures
python -m pip install -r requirements-figures.txt

# Simulation (Python 3.10)
python -m pip install -r requirements-physics.txt
```

## Running the experiment

The stages run in this order. `emg_tst.fetch_gait120` downloads about 16.5 GB. The
simulation runner fetches the MoCapAct reference bank and experts, about 134 GB.

```bash
python -m emg_tst.fetch_gait120 --help
python -m emg_tst.preprocess_gait120 --help
python -m emg_tst.run_gait120_residual_fusion --help
python -m emg_tst.run_gait120_semg_controls --help
python -m emg_tst.run_gait120_temporal_control --help
python -m emg_tst.run_gait120_kinematic_input_check --help
python -m emg_tst.run_gait120_training_path --help
python -m emg_tst.prepare_gait120_physics_panel --help
python -m mocap_phys_eval.run_gait120_residual_fusion --help
python -m mocap_phys_eval.run_gait120_revision_checks --help
python -m emg_tst.run_gait120_trial4_predictions --help
python -m emg_tst.run_gait120_network_path --help
python -m analysis.gait120_robustness --help
python -m analysis.gait120_index_validation --help
```

Set `MOCAPACT_MODELS_DIR` and `MOCAP_PHYS_EVAL_ARTIFACTS_DIR` before running the
simulation stages.

Statistics can be recomputed from an extracted supplement:

```bash
python -m analysis.gait120_conventional_paired_statistics --evidence <extracted>/results --output summary.json
python -m analysis.gait120_checkpoint_correlation --per-window-csv <extracted>/results/analysis/per_window_rollouts.csv --out-dir out
```

Figures need per-checkpoint prediction arrays and rollout recordings, which the
supplement does not include.

## Tests

```bash
python -m pytest tests
```

## License

MIT; see `LICENSE`.
