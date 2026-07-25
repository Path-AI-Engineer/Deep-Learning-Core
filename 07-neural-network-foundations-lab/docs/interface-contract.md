# Neural Network Explainer Interface Contract

## Ownership

The NumPy engine, trace schema and standalone Streamlit interface all live in
Project 07. The interface imports only public package modules and does not
duplicate dense, activation, loss, backpropagation or SGD calculations.

## User journey

1. Choose a supported 2D dataset.
2. Choose hidden width, activation, initialization, learning rate and seed
   within engine-approved limits.
3. Create an experiment and inspect initial parameters.
4. Advance through forward nodes and inspect input, weight, bias, `z` and `a`.
5. Inspect prediction, target and loss.
6. Advance through backward nodes and inspect gradient direction and magnitude.
7. Apply one SGD update or run bounded training.
8. Inspect history and decision boundary.
9. Request the PyTorch parity summary.
10. Reset and reproduce the same initial state.

## Required states

- **Loading:** a real parity calculation is running.
- **Initial:** the deterministic experiment exists but has no training history.
- **Success:** a calculation or training run completed.
- **Error:** an actionable, non-sensitive message explains invalid input or
  engine failure.

## Safe control limits

| Control | Allowed initial range |
|---|---|
| Hidden units | 2–8 |
| Epochs per request | 1–5,000 |
| Learning rate | 0.0001–1.0 |
| Seed | 0–2,147,483,647 |
| Batch samples | Maximum 1,000 |
| Grid resolution | 10–100 per axis |

The engine's configuration contracts are authoritative for validation.
Streamlit constrains inputs for usability but is not the only protection.

## Visual semantics

- Forward flow uses left-to-right direction.
- Backward flow uses right-to-left direction.
- A selected neuron exposes its exact numeric calculation.
- Weight color may encode sign and width may encode bounded magnitude.
- Gradients use a distinct semantic color and include numeric labels.
- Saturation, divergence and invalid configuration are explicit states.

## Presentation boundary

The UI consumes engine objects and versioned trace dictionaries. It may format
and filter them, but it must never implement dense forward, activation, loss,
backpropagation or SGD formulas independently.
