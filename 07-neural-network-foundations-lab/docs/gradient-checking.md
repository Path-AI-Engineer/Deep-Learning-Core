# Gradient Checking

Analytical backpropagation is compared with central finite differences:

```text
df/dθ ≈ (f(θ + ε) - f(θ - ε)) / (2ε)
```

Defaults:

- `epsilon = 1e-6`;
- absolute tolerance `1e-6`;
- relative tolerance `1e-4`;
- `float64` calculations.

A value passes when either its absolute or relative error is within tolerance.
The checker restores every parameter after perturbation and reruns the
analytical pass after validation.

ReLU is not differentiable at `z = 0`. Parity tests therefore move controlled
examples away from that kink; the engine explicitly defines the local
derivative at zero as `0`.

Gradient checking is intentionally used on small datasets and networks because
its cost grows linearly with the number of parameter values. It is an
independent validation tool, not a training algorithm.
