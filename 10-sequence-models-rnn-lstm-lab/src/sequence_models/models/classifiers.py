from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from sequence_models.contracts import ModelConfig


class StatisticsMLP(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 48, num_classes: int = 6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        features = torch.cat(
            (
                inputs.mean(dim=1),
                inputs.std(dim=1, unbiased=False),
                inputs.amin(dim=1),
                inputs.amax(dim=1),
            ),
            dim=1,
        )
        logits: Tensor = self.network(features)
        return logits


class _RecurrentClassifier(nn.Module):
    recurrent: nn.RNN | nn.LSTM | nn.GRU

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        recurrent_class: type[nn.RNN] | type[nn.LSTM] | type[nn.GRU]
        recurrent_class = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[config.model_type]
        kwargs: dict[str, object] = {
            "input_size": config.input_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "batch_first": True,
            "dropout": 0.0,
            "bidirectional": False,
        }
        if config.model_type == "rnn":
            kwargs["nonlinearity"] = "tanh"
        self.recurrent = recurrent_class(**kwargs)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, config.num_classes)

    def encode(self, inputs: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.config.input_size:
            raise ValueError("inputs must have shape [batch, time, 9]")
        recurrent_input: Tensor | torch.nn.utils.rnn.PackedSequence = inputs
        if lengths is not None:
            recurrent_input = pack_padded_sequence(
                inputs,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
        output, state = self.recurrent(recurrent_input)
        hidden = state[0] if isinstance(state, tuple) else state
        return hidden[-1], hidden

    def forward(self, inputs: Tensor, lengths: Tensor | None = None) -> Tensor:
        representation, _ = self.encode(inputs, lengths)
        logits: Tensor = self.head(self.dropout(representation))
        return logits


class RNNClassifier(_RecurrentClassifier):
    pass


class LSTMClassifier(_RecurrentClassifier):
    pass


class GRUClassifier(_RecurrentClassifier):
    pass


def build_model(config: ModelConfig) -> nn.Module:
    config.validate()
    if config.model_type == "statistics-mlp":
        return StatisticsMLP(config.input_size, max(32, config.hidden_size * 2), config.num_classes)
    model_classes = {"rnn": RNNClassifier, "lstm": LSTMClassifier, "gru": GRUClassifier}
    return model_classes[config.model_type](config)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
