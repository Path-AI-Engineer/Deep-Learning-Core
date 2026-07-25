# Backpropagation Guide

Backpropagation applies the chain rule from the scalar loss toward every input
and parameter.

For the output layer:

```text
dL/da₂  = loss derivative
da₂/dz₂ = output activation derivative
dL/dz₂  = dL/da₂ * da₂/dz₂
dL/dW₂  = a₁ᵀ @ dL/dz₂
dL/db₂  = sum(dL/dz₂)
```

The gradient sent to the hidden layer is:

```text
dL/da₁ = dL/dz₂ @ W₂ᵀ
```

For the hidden layer:

```text
da₁/dz₁ = hidden activation derivative
dL/dz₁  = dL/da₁ * da₁/dz₁
dL/dW₁  = Xᵀ @ dL/dz₁
dL/db₁  = sum(dL/dz₁)
dL/dX   = dL/dz₁ @ W₁ᵀ
```

The loss derivative already averages over the batch, so dense-layer gradients
must not divide by the batch size a second time.

## Trace semantics

- `upstream_gradient` is the gradient received from the next operation.
- `local_gradient` is the derivative of the current activation.
- Their product is the preactivation delta.
- `sample_weights` shows one sample's contribution.
- `batch_weights` is the gradient actually used for the SGD update.

This distinction prevents a visual explanation from confusing an individual
path with the aggregate parameter update.
