"""A small multilayer perceptron with explicit, inspectable state."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_network_foundations.activations import activation, derivative
from neural_network_foundations.contracts import NetworkConfig
from neural_network_foundations.layers import Dense, initialize_parameters
from neural_network_foundations.losses import loss, loss_derivative
from neural_network_foundations.tracing import ExecutionTrace, TraceNode

FloatArray = NDArray[np.float64]


class MLP:
    """A two-layer binary classifier designed for explanation and verification."""

    def __init__(self, config: NetworkConfig, *, seed: int) -> None:
        self.config = config
        self.seed = seed
        rng = np.random.default_rng(seed)
        hidden_weights, hidden_bias = initialize_parameters(
            config.input_features,
            config.hidden_units,
            strategy=config.initialization,
            rng=rng,
        )
        output_weights, output_bias = initialize_parameters(
            config.hidden_units,
            config.output_units,
            strategy="xavier" if config.initialization == "he" else config.initialization,
            rng=rng,
        )
        self.hidden = Dense(
            config.input_features,
            config.hidden_units,
            hidden_weights,
            hidden_bias,
            "hidden",
        )
        self.output = Dense(
            config.hidden_units,
            config.output_units,
            output_weights,
            output_bias,
            "output",
        )
        self.hidden_z: FloatArray | None = None
        self.hidden_activation: FloatArray | None = None
        self.output_z: FloatArray | None = None
        self.predictions: FloatArray | None = None
        self.output_upstream: FloatArray | None = None
        self.output_local_gradient: FloatArray | None = None
        self.hidden_upstream: FloatArray | None = None
        self.hidden_local_gradient: FloatArray | None = None
        self.input_gradient: FloatArray | None = None

    def forward(self, features: FloatArray, *, cache: bool = True) -> FloatArray:
        hidden_z = self.hidden.forward(features, cache=cache)
        hidden_activation = activation(self.config.hidden_activation, hidden_z)
        output_z = self.output.forward(hidden_activation, cache=cache)
        predictions = activation(self.config.output_activation, output_z)
        if cache:
            self.hidden_z = hidden_z
            self.hidden_activation = hidden_activation
            self.output_z = output_z
            self.predictions = predictions
        return predictions

    def calculate_loss(self, targets: FloatArray, *, loss_name: str) -> float:
        if self.predictions is None:
            raise RuntimeError("forward must run before calculating loss.")
        return loss(loss_name, self.predictions, targets)

    def backward(self, targets: FloatArray, *, loss_name: str) -> dict[str, FloatArray]:
        if (
            self.predictions is None
            or self.output_z is None
            or self.hidden_z is None
            or self.hidden_activation is None
        ):
            raise RuntimeError("forward must run before backward.")
        target_values = np.asarray(targets, dtype=np.float64)
        if target_values.shape != self.predictions.shape:
            raise ValueError(f"targets must have shape {self.predictions.shape}.")
        self.output_upstream = loss_derivative(loss_name, self.predictions, target_values)
        self.output_local_gradient = derivative(self.config.output_activation, self.output_z)
        output_delta = self.output_upstream * self.output_local_gradient
        self.hidden_upstream = self.output.backward(output_delta)
        self.hidden_local_gradient = derivative(self.config.hidden_activation, self.hidden_z)
        hidden_delta = self.hidden_upstream * self.hidden_local_gradient
        self.input_gradient = self.hidden.backward(hidden_delta)
        return self.gradients()

    def parameters(self) -> dict[str, FloatArray]:
        return {
            "hidden.weights": self.hidden.weights,
            "hidden.bias": self.hidden.bias,
            "output.weights": self.output.weights,
            "output.bias": self.output.bias,
        }

    def gradients(self) -> dict[str, FloatArray]:
        values = {
            "hidden.weights": self.hidden.grad_weights,
            "hidden.bias": self.hidden.grad_bias,
            "output.weights": self.output.grad_weights,
            "output.bias": self.output.grad_bias,
        }
        if any(value is None for value in values.values()):
            raise RuntimeError("backward must run before gradients are available.")
        return {name: value for name, value in values.items() if value is not None}

    def parameter_snapshot(self) -> dict[str, list[list[float]]]:
        return {name: values.tolist() for name, values in self.parameters().items()}

    def load_parameters(self, values: dict[str, FloatArray]) -> None:
        expected = self.parameters()
        if set(values) != set(expected):
            raise ValueError(f"Checkpoint parameters must contain {sorted(expected)}.")
        for name, destination in expected.items():
            source = np.asarray(values[name], dtype=np.float64)
            if source.shape != destination.shape:
                raise ValueError(f"{name} must have shape {destination.shape}.")
            destination[...] = source

    def trace_sample(
        self,
        *,
        dataset: str,
        features: FloatArray,
        targets: FloatArray,
        sample_index: int,
        loss_name: str,
        configuration: dict[str, Any],
    ) -> ExecutionTrace:
        if (
            self.hidden_z is None
            or self.hidden_activation is None
            or self.output_z is None
            or self.predictions is None
        ):
            raise RuntimeError("forward must run before tracing.")
        if not 0 <= sample_index < features.shape[0]:
            raise IndexError("sample_index is outside the current batch.")

        nodes: list[TraceNode] = []
        sample = features[sample_index]
        for index in range(self.config.hidden_units):
            parameter_gradients = None
            upstream_gradient = None
            local_gradient = None
            if (
                self.hidden_upstream is not None
                and self.hidden_local_gradient is not None
                and self.hidden.grad_weights is not None
                and self.hidden.grad_bias is not None
            ):
                upstream_gradient = float(self.hidden_upstream[sample_index, index])
                local_gradient = float(self.hidden_local_gradient[sample_index, index])
                delta = upstream_gradient * local_gradient
                parameter_gradients = {
                    "sample_weights": (sample * delta).tolist(),
                    "sample_bias": delta,
                    "batch_weights": self.hidden.grad_weights[:, index].tolist(),
                    "batch_bias": float(self.hidden.grad_bias[0, index]),
                }
            nodes.append(
                TraceNode(
                    layer_id="hidden",
                    neuron_id=f"hidden-{index}",
                    inputs=sample.tolist(),
                    weights=self.hidden.weights[:, index].tolist(),
                    bias=float(self.hidden.bias[0, index]),
                    z=float(self.hidden_z[sample_index, index]),
                    activation_name=self.config.hidden_activation,
                    activation_value=float(self.hidden_activation[sample_index, index]),
                    upstream_gradient=upstream_gradient,
                    local_gradient=local_gradient,
                    parameter_gradients=parameter_gradients,
                )
            )
        output_parameter_gradients = None
        output_upstream = None
        output_local = None
        if (
            self.output_upstream is not None
            and self.output_local_gradient is not None
            and self.output.grad_weights is not None
            and self.output.grad_bias is not None
        ):
            output_upstream = float(self.output_upstream[sample_index, 0])
            output_local = float(self.output_local_gradient[sample_index, 0])
            output_delta = output_upstream * output_local
            output_parameter_gradients = {
                "sample_weights": (self.hidden_activation[sample_index] * output_delta).tolist(),
                "sample_bias": output_delta,
                "batch_weights": self.output.grad_weights[:, 0].tolist(),
                "batch_bias": float(self.output.grad_bias[0, 0]),
            }
        nodes.append(
            TraceNode(
                layer_id="output",
                neuron_id="output-0",
                inputs=self.hidden_activation[sample_index].tolist(),
                weights=self.output.weights[:, 0].tolist(),
                bias=float(self.output.bias[0, 0]),
                z=float(self.output_z[sample_index, 0]),
                activation_name=self.config.output_activation,
                activation_value=float(self.predictions[sample_index, 0]),
                upstream_gradient=output_upstream,
                local_gradient=output_local,
                parameter_gradients=output_parameter_gradients,
            )
        )
        target = float(targets[sample_index, 0])
        sample_prediction = self.predictions[sample_index : sample_index + 1]
        sample_target = targets[sample_index : sample_index + 1]
        return ExecutionTrace(
            dataset=dataset,
            sample_index=sample_index,
            target=target,
            prediction=float(sample_prediction[0, 0]),
            loss_name=loss_name,
            loss=loss(loss_name, sample_prediction, sample_target),
            nodes=nodes,
            configuration=configuration,
        )
