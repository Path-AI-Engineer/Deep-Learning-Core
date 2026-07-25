"""Equivalent PyTorch computation used only as an independent reference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_network_foundations.losses import loss
from neural_network_foundations.models.mlp import MLP

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ParityReport:
    tolerance: float
    forward_max_absolute_error: float
    loss_absolute_error: float
    gradient_max_absolute_error: dict[str, float]
    input_gradient_max_absolute_error: float
    update_max_absolute_error: dict[str, float]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_with_pytorch(
    model: MLP,
    features: FloatArray,
    targets: FloatArray,
    *,
    loss_name: str,
    learning_rate: float,
    tolerance: float = 1e-8,
) -> ParityReport:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for parity validation.") from exc

    torch.set_default_dtype(torch.float64)
    input_tensor = torch.tensor(features, dtype=torch.float64, requires_grad=True)
    target_tensor = torch.tensor(targets, dtype=torch.float64)
    hidden = torch.nn.Linear(
        model.config.input_features, model.config.hidden_units, dtype=torch.float64
    )
    output = torch.nn.Linear(
        model.config.hidden_units, model.config.output_units, dtype=torch.float64
    )
    with torch.no_grad():
        hidden.weight.copy_(torch.tensor(model.hidden.weights.T))
        hidden.bias.copy_(torch.tensor(model.hidden.bias.reshape(-1)))
        output.weight.copy_(torch.tensor(model.output.weights.T))
        output.bias.copy_(torch.tensor(model.output.bias.reshape(-1)))

    hidden_z = hidden(input_tensor)
    if model.config.hidden_activation == "sigmoid":
        hidden_activation = torch.sigmoid(hidden_z)
    elif model.config.hidden_activation == "tanh":
        hidden_activation = torch.tanh(hidden_z)
    elif model.config.hidden_activation == "relu":
        hidden_activation = torch.relu(hidden_z)
    else:
        raise ValueError(f"Unsupported activation: {model.config.hidden_activation}.")
    torch_predictions = torch.sigmoid(output(hidden_activation))
    if loss_name == "binary_cross_entropy":
        torch_loss = torch.nn.functional.binary_cross_entropy(torch_predictions, target_tensor)
    elif loss_name == "mean_squared_error":
        torch_loss = torch.nn.functional.mse_loss(torch_predictions, target_tensor)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}.")
    torch_loss.backward()

    numpy_predictions = model.forward(features)
    numpy_loss = loss(loss_name, numpy_predictions, targets)
    numpy_gradients = model.backward(targets, loss_name=loss_name)
    parameter_gradients = (
        hidden.weight.grad,
        hidden.bias.grad,
        output.weight.grad,
        output.bias.grad,
    )
    if any(gradient is None for gradient in parameter_gradients):
        raise RuntimeError("PyTorch did not produce all parameter gradients.")
    hidden_weight_gradient = parameter_gradients[0]
    hidden_bias_gradient = parameter_gradients[1]
    output_weight_gradient = parameter_gradients[2]
    output_bias_gradient = parameter_gradients[3]
    assert hidden_weight_gradient is not None
    assert hidden_bias_gradient is not None
    assert output_weight_gradient is not None
    assert output_bias_gradient is not None
    torch_gradients = {
        "hidden.weights": hidden_weight_gradient.detach().numpy().T,
        "hidden.bias": hidden_bias_gradient.detach().numpy().reshape(1, -1),
        "output.weights": output_weight_gradient.detach().numpy().T,
        "output.bias": output_bias_gradient.detach().numpy().reshape(1, -1),
    }
    gradient_errors = {
        name: float(np.max(np.abs(numpy_gradients[name] - torch_gradients[name])))
        for name in numpy_gradients
    }
    update_errors: dict[str, float] = {}
    for name, parameter in model.parameters().items():
        numpy_updated = parameter - learning_rate * numpy_gradients[name]
        if name == "hidden.weights":
            torch_parameter = hidden.weight.detach().numpy().T
        elif name == "hidden.bias":
            torch_parameter = hidden.bias.detach().numpy().reshape(1, -1)
        elif name == "output.weights":
            torch_parameter = output.weight.detach().numpy().T
        else:
            torch_parameter = output.bias.detach().numpy().reshape(1, -1)
        torch_updated = torch_parameter - learning_rate * torch_gradients[name]
        update_errors[name] = float(np.max(np.abs(numpy_updated - torch_updated)))

    forward_error = float(np.max(np.abs(numpy_predictions - torch_predictions.detach().numpy())))
    loss_error = abs(numpy_loss - float(torch_loss.detach().numpy()))
    if model.input_gradient is None or input_tensor.grad is None:
        raise RuntimeError("Input gradients were not produced.")
    input_gradient_error = float(
        np.max(np.abs(model.input_gradient - input_tensor.grad.detach().numpy()))
    )
    all_errors = [
        forward_error,
        loss_error,
        input_gradient_error,
        *gradient_errors.values(),
        *update_errors.values(),
    ]
    return ParityReport(
        tolerance=tolerance,
        forward_max_absolute_error=forward_error,
        loss_absolute_error=loss_error,
        gradient_max_absolute_error=gradient_errors,
        input_gradient_max_absolute_error=input_gradient_error,
        update_max_absolute_error=update_errors,
        passed=all(value <= tolerance for value in all_errors),
    )
