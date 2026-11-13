# Analysis protocol summary

Summary of the Gait120 residual-fusion and paired simulation experiment.

## Prediction experiment

- Dataset: Gait120 level walking only.
- Development cohort: S001–S030. Trials 1–3 supplied scaling values and pooled
  fitting data, trial 4 selected hyperparameters, and trial 5 was held out.
- Confirmation cohort: S031–S120. The fixed model form and hyperparameters were
  refitted with pooled trials 1–3, trial 4 was secondary, and trial 5 was the
  primary test.
- Model fitting used equal total sample weight per participant. This is a
  later-trial, participant-calibrated design, not transfer to uncalibrated
  participants.
- Kinematic input: 60 recorded knee-angle frames (600 ms at 100 Hz).
- sEMG input: the final 15 recorded causal envelope frames from 12 right-leg
  channels (150 ms at 100 Hz).
- Forecast target: knee angle 100 ms after the final input frame.
- No predictor input or target was interpolated or time-normalized.
- Baseline: ridge regression using knee-angle history.
- Residual fusion: a second ridge regression estimated the remaining training
  residual from sEMG; the ablation removed only this correction.
- Primary statistical unit: participant.
- Primary summary: two-sided paired t-test and Student-t confidence interval on
  participant RMSE without sEMG minus RMSE with residual fusion. Wilcoxon and
  exact sign tests were sensitivity analyses.

### sEMG negative controls

Three surrogate conditions repeat the whole confirmation fit with the sEMG replaced
by a signal that keeps some property of the recording and breaks its correspondence
with the knee:

- `circular_shift`: each participant-trial sEMG block is rotated against its own
  kinematics, preserving amplitudes, spectra, and cross-channel structure.
- `participant_swap`: each participant receives another participant's sEMG.
- `phase_randomized`: one random phase per frequency is applied to all twelve
  channels, preserving each power spectrum and the phase relationships between
  channels.

The kinematic stage is shared across all four conditions, so any difference is
attributable to sEMG content.

Attribution is judged by a paired within-participant contrast: for each
participant, the improvement with recorded sEMG minus the improvement with that
surrogate, summarized with the same bootstrap interval and sign-flip
randomization test as the primary ablation. The improvement is attributed to
sEMG content when the recorded signal clears its gate and beats every surrogate
on that contrast.

## Timing control

The same sEMG history was shifted 500 ms earlier within each continuous trial.
Rows without a complete earlier history were excluded, and aligned and shifted
models were evaluated on identical target rows.

## Paired simulation

- Fixed panel: 80 one-second confirmation trial-5 windows selected in seeded,
  participant-balanced rounds before motion matching or simulation.
- Reference bank: the whole MoCapAct expert zoo, roughly 2589 snippets. The reported
  run matched the 80 windows to 32 distinct snippets: 51 windows on walking clips, 26
  on basketball forward dribbling, and 3 on a mixed-activity or undescribed clip, by
  the CMU database's descriptions.
- Matching: knee-angle RMSE ranked candidates. Right-thigh pitch RMSE was
  calculated afterward as a match-quality covariate. No amplitude scaling was
  allowed.
- Prediction mapping: the predicted knee angle is the tracking target of the PD
  override on the right knee actuator, mapped into the matched clip's knee convention
  by the constant offset and sign the match established.
- Paired conditions shared the reference snippet, initial state, expert policy, and
  warm-up. Only the prediction condition retunes the knee actuator.
- Accuracy range: 13 finite-step checkpoints along the gradient-descent path of the
  same model, from a zero initialization that predicts the participant mean, plus the
  closed-form confirmation fit as a separate terminal checkpoint. Checkpoints tile the
  achieved training RMSE. Every checkpoint used the same 80 windows and references.
- Zero-error replay: each window's matched clip is commanded with its own recorded
  knee trajectory through the override.
- Alternate gains: the panel and the zero-error replay are repeated at Kp = 800,
  Kd = 40.

## Simulation outcomes

The primary outcome was excess-instability area under the curve (AUC): the
prediction-condition value of a study-specific XCoM-based index minus the
matched unmodified-reference value. The bounded index has not been validated
as a clinical stability or fall-risk measure.

The adjusted association analysis removed ranked knee and thigh matching
errors before calculating the remaining participant-level Spearman
association between prediction RMSE and excess AUC. A high-instability flag was
reported only as a descriptive algorithmic outcome; it was not an observed
fall.

All analyses along the training path are exploratory.

Three association views are reported. `per_checkpoint` repeats the adjusted
analysis at each accuracy level and reports the RMSE spread with it. `within_window` correlates each window's own accuracy against its own
excess instability across checkpoints and combines the results with a Fisher-z
one-sample test, which removes matched motion, snippet, and initial state as
sources of variation. `pooled` uses every checkpoint-window pair with a
participant-level cluster bootstrap, since each window recurs once per
checkpoint and several participants contribute two windows.

The two-segment fit of mean excess instability against mean RMSE is descriptive and
was chosen after plotting the checkpoint means.

## Replication analyses

- Independent tuning: ten windows from ten development-cohort participants, matched to
  clips outside the analysis panel, scored at the same five gain settings.
- Second panel: 80 windows drawn from confirmation trial 4 by the panel procedure,
  matched afresh, and replayed at the same checkpoints and gains.
- Network models: a one-layer GRU and LSTM (32 units), a two-layer perceptron (64 units),
  a four-layer dilated convolutional network (32 channels), and a two-layer transformer
  encoder (32 dimensions, 4 heads), on the same inputs, trained by Adam with
  participant-balanced sampling and no hyperparameter tuning. Checkpoints are taken where
  training RMSE first reaches that of each ridge checkpoint, once per distinct model, and each
  is replayed on the reported panel, matches, and gains. The GRU was run first, the LSTM
  second, and the other three after both results.
- Pooled decline: the slope below the split averaged over the ridge model and the five
  networks, with one participant bootstrap shared by all six.
- Index checks: the index recomputed under alternative definitions, outcomes that do not
  use the XCoM, falls, and an upper bound on early termination.

## Robustness analyses

The two-segment fit and the within-window analysis are repeated on walking clips,
without basketball clips, without the ten gain-tuning windows, without windows whose
zero-error replay flagged a problem, and at the alternate gains. The fit is also
repeated resampling matched clips instead of participants.
