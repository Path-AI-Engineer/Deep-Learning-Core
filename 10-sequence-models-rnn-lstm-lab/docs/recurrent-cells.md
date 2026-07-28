# Recurrent Cells

A vanilla RNN repeatedly combines current input and previous hidden state through a tanh transformation. It is compact, but long chains of Jacobian products can erase or amplify gradients.

LSTM adds input, forget, candidate, and output gates plus a separate cell state. Its additive memory path improves control over long dependencies at additional parameter cost.

GRU combines reset and update behavior without a separate cell state. It is smaller than LSTM and often a strong practical alternative.

`sequence_models.cells` reproduces one-step equations with NumPy and verifies final-state parity against PyTorch `RNNCell`, `LSTMCell`, and `GRUCell` within `1e-9`.

