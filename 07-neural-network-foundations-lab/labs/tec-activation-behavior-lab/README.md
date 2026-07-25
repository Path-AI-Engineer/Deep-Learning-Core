# Activation Behavior Lab

The lab compares sigmoid, tanh and ReLU over controlled negative, near-zero and
positive inputs.

- Sigmoid compresses values into `(0, 1)` and saturates at both extremes.
- Tanh centers output around zero and also saturates at extremes.
- ReLU preserves positive values but has a zero local gradient for negative
  preactivations.

`tests/unit/test_activations.py` compares every analytical derivative against a
central finite difference away from ReLU's nondifferentiable origin.
