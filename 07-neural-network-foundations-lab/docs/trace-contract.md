# Execution Trace Contract v1

The trace is the only source of official neural-network calculations for the
visual interface.

The machine-readable contract is versioned at
[`contracts/trace-v1.schema.json`](../contracts/trace-v1.schema.json). The
Markdown document explains semantics; the JSON Schema protects structure.

## Envelope

| Field | Meaning |
|---|---|
| `schema_version` | Contract version, currently `1.0` |
| `generated_at` | UTC timestamp |
| `dataset` | Stable catalog identifier |
| `sample_index` | Traced sample inside the batch |
| `target` | Observed binary label |
| `prediction` | Model output |
| `loss_name` / `loss` | Objective and scalar sample loss |
| `configuration` | Complete validated experiment configuration |
| `nodes` | Ordered hidden and output neuron traces |

## Node

Every node exposes:

- `layer_id` and `neuron_id`;
- input activation values;
- incoming weights and bias;
- preactivation `z`;
- activation name and value;
- optional upstream and local gradients;
- optional weight and bias gradients.

Gradient fields are `null` for a forward-only trace and populated after
backpropagation. Consumers must tolerate additive fields but reject unknown
major schema versions.

## Formula

For each node:

```text
z = sum(inputs[i] * weights[i]) + bias
activation_value = activation(z)
```

The frontend may display this equation using supplied values but may not replace
the engine result with a separately calculated official result.

## Example

Run:

```powershell
python scripts/export_demo_trace.py --dataset xor --sample-index 1
```

The generated `artifacts/traces/demo-forward-trace.json` is a real engine
artifact and can be regenerated from the recorded seed and configuration.
