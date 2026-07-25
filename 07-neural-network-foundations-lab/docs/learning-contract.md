# Learning Contract

## Purpose

This lab teaches the mechanics of a neural network through inspectable,
reproducible calculations. A learner must connect every visual value to a
mathematical operation performed by the engine.

## Primary learner

A developer familiar with classical machine learning who wants to understand
deep-learning fundamentals before delegating differentiation and optimization
to a framework.

## Questions the lab must answer

| Concept | Observable answer |
|---|---|
| Input | Feature values, tensor shape and dataset meaning |
| Weight | Contribution of an input to a neuron's preactivation |
| Bias | Offset added independently of the input |
| Dense layer | `z = xW + b`, with dimensions and values |
| Activation | Transformation from `z` to `a` and local derivative |
| Prediction | Output value and binary interpretation |
| Loss | Scalar error and the inputs used to calculate it |
| Backpropagation | Upstream, local and parameter gradients |
| SGD | Old value, gradient, learning rate and updated value |
| Training | Loss/accuracy history and bounded progress |
| Reproducibility | Dataset, configuration, seed and checkpoint |
| Framework parity | Absolute differences against PyTorch |

## Completion evidence

The learning goal is met only when:

1. known forward calculations pass exact or tolerance-based tests;
2. analytical gradients match central finite differences;
3. XOR is solved by a hidden-layer MLP under a recorded configuration;
4. repeating a seed reproduces parameters and histories;
5. a saved checkpoint restores the same prediction;
6. the public trace can be rendered without importing engine internals;
7. the NumPy implementation matches the equivalent PyTorch computation.

Decreasing loss alone is insufficient evidence.

## Pedagogical boundaries

- Network size is intentionally small enough to inspect.
- Controls are bounded to protect CPU time and visual clarity.
- Numerical values remain visible; animations never replace evidence.
- The frontend explains engine outputs but never recalculates official math.
- PyTorch is introduced only after the manual implementation is validated.
