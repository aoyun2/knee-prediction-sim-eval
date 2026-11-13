from __future__ import annotations

import unittest

import numpy as np

import emg_tst.gait120_experiment as gait
import emg_tst.gait120_training_path as tpath
import emg_tst.run_gait120_residual_fusion as fusion


def _synthetic_examples(
    *, n_subjects: int = 6, n_frames: int = 40, seed: int = 3
) -> gait.ExampleSet:
    """Small example set with the shape the residual-fusion features expect.

    The knee channel is a smooth oscillation so the kinematic stage has real
    autoregressive structure, and the sEMG channels carry a lagged copy of the
    forecast error so the residual stage has something to find.
    """
    rng = np.random.default_rng(seed)
    frames = fusion.KINEMATIC_FRAMES
    n_emg = gait.N_EMG

    x_rows: list[np.ndarray] = []
    y_std: list[float] = []
    subjects: list[int] = []
    trials: list[int] = []
    ends: list[int] = []
    targets: list[int] = []

    for subject in range(1, n_subjects + 1):
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        for trial in (1, 2):
            for frame in range(n_frames):
                t = np.arange(frames, dtype=np.float64) * 0.1 + frame * 0.1 + phase
                knee = np.sin(t) + 0.05 * rng.standard_normal(frames)
                future = float(np.sin(t[-1] + 1.0))
                emg = 0.3 * np.cos(t)[:, None] * np.ones((1, n_emg))
                emg = emg + 0.05 * rng.standard_normal((frames, n_emg))
                block = np.concatenate([emg, knee[:, None]], axis=1)
                x_rows.append(block.astype(np.float32))
                y_std.append(future)
                subjects.append(subject)
                trials.append(trial)
                ends.append(frame + frames)
                targets.append(frame + frames + 10)

    x = np.stack(x_rows).astype(np.float32)
    y_standardized = np.asarray(y_std, dtype=np.float64)
    target_mean = np.full(y_standardized.shape, 170.0, dtype=np.float64)
    target_std = np.full(y_standardized.shape, 12.0, dtype=np.float64)
    return gait.ExampleSet(
        x=x,
        y_standardized=y_standardized,
        y_deg=(y_standardized * target_std + target_mean).astype(np.float32),
        target_mean_deg=target_mean,
        target_std_deg=target_std,
        subject_number=np.asarray(subjects, dtype=np.int16),
        trial_index=np.asarray(trials, dtype=np.int16),
        input_end_frame=np.asarray(ends, dtype=np.int32),
        target_frame=np.asarray(targets, dtype=np.int32),
    )


class TrainingPathTests(unittest.TestCase):
    def setUp(self) -> None:
        gait.INPUT_FRAMES = fusion.KINEMATIC_FRAMES
        self.train = _synthetic_examples()
        self.path = tpath.fit_training_path(
            self.train,
            kinematic_alpha=0.1,
            residual_alpha=10.0,
            max_steps=40_000,
            n_checkpoints=10,
        )

    def test_descent_reaches_the_closed_form_confirmation_fit(self) -> None:
        """The converged checkpoint is the closed-form confirmation model."""
        kinematic, residual, caps = tpath.materialize_checkpoint(
            self.path, int(self.path.checkpoint_indices.size) - 1, self.train
        )
        exact_k, exact_e, exact_caps = fusion._fit_fixed(
            self.train, kinematic_alpha=0.1, residual_alpha=10.0
        )
        np.testing.assert_allclose(
            kinematic.coefficient, exact_k.coefficient, rtol=1.0e-3, atol=1.0e-5
        )
        np.testing.assert_allclose(
            residual.coefficient, exact_e.coefficient, rtol=1.0e-3, atol=1.0e-5
        )
        for subject, cap in exact_caps.items():
            self.assertAlmostEqual(caps[subject], cap, places=5)

    def test_path_starts_at_the_participant_mean_and_improves(self) -> None:
        """Descending from zero spans a wide accuracy range."""
        first = float(self.path.train_rmse_deg[0])
        last = float(self.path.train_rmse_deg[-1])
        self.assertGreater(first, last)
        # Step zero predicts the target mean, so its error is the target spread.
        expected = float(np.std(np.asarray(self.train.y_deg, dtype=np.float64)))
        self.assertAlmostEqual(first, expected, delta=0.05 * expected)

        # Any transient rise in RMSE is small against the span.
        step_change = np.diff(self.path.train_rmse_deg)
        span = first - last
        self.assertLess(float(np.max(step_change)), 0.01 * span)
        self.assertGreater(span, 0.5 * first)

    def test_rmse_ladder_spreads_checkpoints_more_evenly_than_step_index(self) -> None:
        """The RMSE ladder spreads checkpoints across the achieved range."""
        rmse = self.path.train_rmse_deg
        ladder = rmse[tpath.select_checkpoints(self.path, mode="rmse_ladder", count=10)]
        by_step = rmse[tpath.select_checkpoints(self.path, mode="log_step", count=10)]
        span = float(rmse[0] - rmse[-1])

        def worst_gap(values: np.ndarray) -> float:
            return float(np.max(np.abs(np.diff(np.sort(values))))) / span

        self.assertLess(worst_gap(ladder), worst_gap(by_step))
        self.assertLess(worst_gap(ladder), 0.5)

    def test_checkpoints_include_both_ends_and_are_ordered(self) -> None:
        picks = tpath.select_checkpoints(self.path, count=8)
        self.assertEqual(int(picks[0]), 0)
        self.assertEqual(int(picks[-1]), int(self.path.steps.size) - 1)
        self.assertTrue(np.all(np.diff(picks) > 0))

    def test_unknown_checkpoint_mode_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            tpath.select_checkpoints(self.path, mode="whatever")


if __name__ == "__main__":
    unittest.main()
