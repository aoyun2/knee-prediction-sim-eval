"""Train a neural knee-angle predictor and export predictions along its training path.

A one-layer GRU or LSTM, a two-layer perceptron, a dilated convolutional network, or a
two-layer transformer encoder reads the same 60 knee frames and final 15 sEMG frames as
the residual-fusion model and is fitted to the same participant-balanced trials 1-3 from
a zero output layer. Checkpoints are taken where its training RMSE first reaches the
training RMSE of each residual-fusion checkpoint; a model that reaches several at once is
kept once. Outputs use the layout
``prepare_gait120_physics_panel`` reads. Runs in the simulation environment.

    python -m emg_tst.run_gait120_network_path --cell lstm --cache-dir <cache> \
        --reference-path-dir <path> --run-dir <out>
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import emg_tst.gait120_experiment as gait
import emg_tst.gait120_training_path as tpath
import emg_tst.run_gait120_residual_fusion as fusion

SEED = 20261105
HIDDEN = 32
LEARNING_RATE = 3.0e-4
BATCH = 1024
MAX_STEPS = 40_000
EVAL_ROWS = 20_000
PREDICT_BATCH = 8192
CHANNELS = gait.N_EMG + 1
FRAMES = fusion.KINEMATIC_FRAMES


def _zero_head(width: int) -> torch.nn.Linear:
    head = torch.nn.Linear(width, 1)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return head


class RecurrentPredictor(torch.nn.Module):
    def __init__(self, cell: str = "gru") -> None:
        super().__init__()
        self.gru = {"gru": torch.nn.GRU, "lstm": torch.nn.LSTM}[cell](
            CHANNELS, HIDDEN, batch_first=True
        )
        self.head = _zero_head(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gru(x)[0][:, -1]).squeeze(-1)


class PerceptronPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(FRAMES * CHANNELS, 2 * HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * HIDDEN, 2 * HIDDEN),
            torch.nn.ReLU(),
        )
        self.head = _zero_head(2 * HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


class ConvolutionalPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers, width = [], CHANNELS
        for dilation in (1, 2, 4, 8):
            layers += [
                torch.nn.Conv1d(width, HIDDEN, kernel_size=3, dilation=dilation, padding=dilation),
                torch.nn.ReLU(),
            ]
            width = HIDDEN
        self.body = torch.nn.Sequential(*layers)
        self.head = _zero_head(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x.transpose(1, 2))[:, :, -1]).squeeze(-1)


class TransformerPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(CHANNELS, HIDDEN)
        self.position = torch.nn.Parameter(torch.zeros(1, FRAMES, HIDDEN))
        layer = torch.nn.TransformerEncoderLayer(
            HIDDEN, nhead=4, dim_feedforward=2 * HIDDEN, dropout=0.0, batch_first=True
        )
        self.body = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.head = _zero_head(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.embed(x) + self.position)[:, -1]).squeeze(-1)


def _build(cell: str) -> torch.nn.Module:
    if cell in ("gru", "lstm"):
        return RecurrentPredictor(cell)
    return {"mlp": PerceptronPredictor, "cnn": ConvolutionalPredictor,
            "transformer": TransformerPredictor}[cell]()


CELLS = ("gru", "lstm", "mlp", "cnn", "transformer")
DESCRIPTIONS = {
    "gru": "one-layer GRU with a linear output",
    "lstm": "one-layer LSTM with a linear output",
    "mlp": "two-layer perceptron (64 units per layer) with a linear output",
    "cnn": "four-layer dilated 1-D convolutional network (32 channels) with a linear output",
    "transformer": "two-layer transformer encoder (32 dimensions, 4 heads) with a linear output",
}


def _inputs(examples: gait.ExampleSet, use_emg: bool = True) -> torch.Tensor:
    x = np.array(examples.x, dtype=np.float32)
    x[:, : -fusion.EMG_FRAMES, : gait.N_EMG] = 0.0
    if not use_emg:
        x[:, :, : gait.N_EMG] = 0.0
    return torch.from_numpy(x)


@torch.no_grad()
def _predict(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    return np.concatenate(
        [model(x[i : i + PREDICT_BATCH]).numpy() for i in range(0, x.shape[0], PREDICT_BATCH)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-path-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cell", choices=CELLS, default="gru")
    args = parser.parse_args()

    reference = json.loads(
        (args.reference_path_dir / "checkpoints" / "manifest.json").read_text(encoding="utf-8")
    )
    targets = [float(row["train_rmse_deg"]) for row in reference["checkpoints"]]

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    gait.INPUT_FRAMES = fusion.KINEMATIC_FRAMES
    train, _, test, _ = gait._build_examples(args.cache_dir.resolve(), list(range(31, 121)))
    x = _inputs(train)
    y = torch.from_numpy(np.asarray(train.y_standardized, dtype=np.float32))
    scale = np.asarray(train.target_std_deg, dtype=np.float64)
    weights = np.asarray(fusion._participant_balanced_weights(train.subject_number), dtype=np.float64)
    weights = weights / weights.sum()
    eval_rows = rng.choice(len(train), size=min(EVAL_ROWS, len(train)), replace=False)

    def training_rmse(model: torch.nn.Module) -> float:
        error = (_predict(model, x[eval_rows]) - y[eval_rows].numpy()) * scale[eval_rows]
        return float(np.sqrt(np.mean(error**2)))

    model = _build(args.cell)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    captured: list[tuple[int, float, dict]] = []
    step = 0
    rmse = training_rmse(model)
    captured.append((0, rmse, copy.deepcopy(model.state_dict())))
    next_target = 1
    while next_target < len(targets) and step < MAX_STEPS:
        batch = torch.from_numpy(rng.choice(len(train), size=BATCH, p=weights))
        optimizer.zero_grad()
        loss = torch.mean((model(x[batch]) - y[batch]) ** 2)
        loss.backward()
        optimizer.step()
        step += 1
        if step <= 2000 and step % 5 == 0 or step % 50 == 0:
            rmse = training_rmse(model)
            reached = next_target
            while next_target < len(targets) and rmse <= targets[next_target]:
                next_target += 1
            if next_target > reached:
                captured.append((step, rmse, copy.deepcopy(model.state_dict())))
                print(f"step {step}: train RMSE {rmse:.3f} reached rung {next_target - 1}", flush=True)
    if next_target < len(targets):
        captured.append((step, training_rmse(model), copy.deepcopy(model.state_dict())))

    x_test, x_test_no_emg = _inputs(test), _inputs(test, use_emg=False)
    target_std = np.asarray(test.target_std_deg, dtype=np.float64)
    target_mean = np.asarray(test.target_mean_deg, dtype=np.float64)
    labels, rows = [], []
    for order, (at_step, rmse, state) in enumerate(captured):
        label = tpath.checkpoint_label(order, at_step)
        labels.append(label)
        model.load_state_dict(state)
        fused = _predict(model, x_test) * target_std + target_mean
        no_emg = _predict(model, x_test_no_emg) * target_std + target_mean
        gait._atomic_npz(
            args.run_dir / "checkpoints" / label / "test_predictions.npz",
            target_deg=np.asarray(test.y_deg, dtype=np.float32),
            subject_number=np.asarray(test.subject_number, dtype=np.int16),
            trial_index=np.asarray(test.trial_index, dtype=np.int16),
            input_end_frame=np.asarray(test.input_end_frame, dtype=np.int32),
            target_frame=np.asarray(test.target_frame, dtype=np.int32),
            fused_prediction_deg=np.asarray(fused, dtype=np.float32),
            no_emg_prediction_deg=np.asarray(no_emg, dtype=np.float32),
        )
        metrics = gait._participant_metrics(test, fused)
        no_emg_metrics = gait._participant_metrics(test, no_emg)
        rows.append(
            {
                "label": label,
                "step": at_step,
                "train_rmse_deg": rmse,
                "fused_mean_participant_rmse_deg": metrics["mean_participant_rmse_deg"],
                "no_emg_mean_participant_rmse_deg": no_emg_metrics["mean_participant_rmse_deg"],
            }
        )
        print(
            label,
            f"train {rmse:.2f}",
            f"test {rows[-1]['fused_mean_participant_rmse_deg']:.3f}",
            f"no sEMG {rows[-1]['no_emg_mean_participant_rmse_deg']:.3f}",
            flush=True,
        )

    gait._atomic_json(
        args.run_dir / "checkpoints" / "manifest.json",
        {"labels": labels, "primary_checkpoint": labels[-1], "checkpoints": rows},
    )
    gait._atomic_json(
        args.run_dir / "protocol.json",
        {
            "model": DESCRIPTIONS[args.cell],
            "cell": args.cell,
            "seed": SEED,
            "hidden_units": HIDDEN,
            "learning_rate": LEARNING_RATE,
            "batch": BATCH,
            "max_steps": MAX_STEPS,
            "final_step": step,
            "training_rmse_rows": int(eval_rows.size),
            "torch": torch.__version__,
        },
    )


if __name__ == "__main__":
    main()
