from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _sigmoid(values: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-values))


def _validate_step(
    x: FloatArray,
    h: FloatArray,
    weight_ih: FloatArray,
    weight_hh: FloatArray,
    gates: int,
) -> None:
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("x and h must be one-dimensional vectors")
    if weight_ih.shape != (gates * h.size, x.size):
        raise ValueError("weight_ih has an incompatible shape")
    if weight_hh.shape != (gates * h.size, h.size):
        raise ValueError("weight_hh has an incompatible shape")


def rnn_step(
    x: FloatArray,
    h: FloatArray,
    weight_ih: FloatArray,
    weight_hh: FloatArray,
    bias_ih: FloatArray,
    bias_hh: FloatArray,
) -> FloatArray:
    _validate_step(x, h, weight_ih, weight_hh, 1)
    return np.asarray(
        np.tanh(weight_ih @ x + bias_ih + weight_hh @ h + bias_hh),
        dtype=np.float64,
    )


def lstm_step(
    x: FloatArray,
    h: FloatArray,
    c: FloatArray,
    weight_ih: FloatArray,
    weight_hh: FloatArray,
    bias_ih: FloatArray,
    bias_hh: FloatArray,
) -> tuple[FloatArray, FloatArray, dict[str, FloatArray]]:
    _validate_step(x, h, weight_ih, weight_hh, 4)
    if c.shape != h.shape:
        raise ValueError("cell state must have the same shape as hidden state")
    gates = weight_ih @ x + bias_ih + weight_hh @ h + bias_hh
    input_raw, forget_raw, candidate_raw, output_raw = np.split(gates, 4)
    input_gate = _sigmoid(input_raw)
    forget_gate = _sigmoid(forget_raw)
    candidate = np.tanh(candidate_raw)
    output_gate = _sigmoid(output_raw)
    next_c = forget_gate * c + input_gate * candidate
    next_h = output_gate * np.tanh(next_c)
    return next_h, next_c, {
        "input": input_gate,
        "forget": forget_gate,
        "candidate": candidate,
        "output": output_gate,
    }


def gru_step(
    x: FloatArray,
    h: FloatArray,
    weight_ih: FloatArray,
    weight_hh: FloatArray,
    bias_ih: FloatArray,
    bias_hh: FloatArray,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    _validate_step(x, h, weight_ih, weight_hh, 3)
    input_r, input_z, input_n = np.split(weight_ih @ x + bias_ih, 3)
    hidden_r, hidden_z, hidden_n = np.split(weight_hh @ h + bias_hh, 3)
    reset = _sigmoid(input_r + hidden_r)
    update = _sigmoid(input_z + hidden_z)
    new = np.tanh(input_n + reset * hidden_n)
    next_h = (1.0 - update) * new + update * h
    return next_h, {"reset": reset, "update": update, "new": new}


@dataclass(frozen=True, slots=True)
class CellTrace:
    cell_type: str
    timesteps: list[dict[str, object]]
    educational_output: list[float]
    pytorch_output: list[float]
    max_abs_difference: float
    parity_tolerance: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _weights(rows: int, columns: int, start: float) -> FloatArray:
    return np.linspace(start, start + 0.42, rows * columns, dtype=np.float64).reshape(
        rows, columns
    )


def _to_list(values: FloatArray) -> list[float]:
    return [round(float(value), 7) for value in values]


def cell_trace(cell_type: Literal["rnn", "lstm", "gru"]) -> CellTrace:
    input_size, hidden_size = 2, 3
    sequence = np.array([[0.25, -0.4], [0.1, 0.35], [-0.2, 0.45]], dtype=np.float64)
    gate_count = {"rnn": 1, "lstm": 4, "gru": 3}[cell_type]
    weight_ih = _weights(gate_count * hidden_size, input_size, -0.21)
    weight_hh = _weights(gate_count * hidden_size, hidden_size, -0.15)
    bias_ih = np.linspace(-0.05, 0.07, gate_count * hidden_size, dtype=np.float64)
    bias_hh = np.linspace(0.03, -0.04, gate_count * hidden_size, dtype=np.float64)
    hidden = np.array([0.08, -0.03, 0.04], dtype=np.float64)
    cell = np.array([0.02, 0.01, -0.06], dtype=np.float64)
    initial_hidden = hidden.copy()
    initial_cell = cell.copy()
    steps: list[dict[str, object]] = []

    for index, values in enumerate(sequence):
        previous_hidden = hidden.copy()
        step: dict[str, object] = {
            "timestep": index,
            "input": _to_list(values),
            "previous_hidden": _to_list(previous_hidden),
        }
        if cell_type == "rnn":
            hidden = rnn_step(
                values, hidden, weight_ih, weight_hh, bias_ih, bias_hh
            )
        elif cell_type == "lstm":
            previous_cell = cell.copy()
            hidden, cell, gates = lstm_step(
                values, hidden, cell, weight_ih, weight_hh, bias_ih, bias_hh
            )
            step["previous_cell"] = _to_list(previous_cell)
            step["cell"] = _to_list(cell)
            step["gates"] = {name: _to_list(gate) for name, gate in gates.items()}
        else:
            hidden, gates = gru_step(
                values, hidden, weight_ih, weight_hh, bias_ih, bias_hh
            )
            step["gates"] = {name: _to_list(gate) for name, gate in gates.items()}
        step["hidden"] = _to_list(hidden)
        step["hidden_norm"] = round(float(np.linalg.norm(hidden)), 7)
        steps.append(step)

    torch.set_default_dtype(torch.float64)
    if cell_type == "rnn":
        torch_cell: torch.nn.RNNCell | torch.nn.LSTMCell | torch.nn.GRUCell
        torch_cell = torch.nn.RNNCell(input_size, hidden_size)
    elif cell_type == "lstm":
        torch_cell = torch.nn.LSTMCell(input_size, hidden_size)
    else:
        torch_cell = torch.nn.GRUCell(input_size, hidden_size)
    with torch.no_grad():
        torch_cell.weight_ih.copy_(torch.from_numpy(weight_ih))
        torch_cell.weight_hh.copy_(torch.from_numpy(weight_hh))
        cast(torch.Tensor, torch_cell.bias_ih).copy_(torch.from_numpy(bias_ih))
        cast(torch.Tensor, torch_cell.bias_hh).copy_(torch.from_numpy(bias_hh))
        torch_hidden = torch.from_numpy(initial_hidden)
        torch_cell_state = torch.from_numpy(initial_cell)
        for values in torch.from_numpy(sequence):
            if cell_type == "lstm":
                torch_hidden, torch_cell_state = torch_cell(
                    values, (torch_hidden, torch_cell_state)
                )
            else:
                torch_hidden = torch_cell(values, torch_hidden)
    torch.set_default_dtype(torch.float32)
    pytorch_output = torch_hidden.detach().numpy()
    difference = float(np.max(np.abs(hidden - pytorch_output)))
    return CellTrace(
        cell_type=cell_type,
        timesteps=steps,
        educational_output=_to_list(hidden),
        pytorch_output=_to_list(pytorch_output),
        max_abs_difference=difference,
        parity_tolerance=1e-9,
    )
