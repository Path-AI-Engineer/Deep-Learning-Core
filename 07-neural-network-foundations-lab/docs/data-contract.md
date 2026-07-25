# Dataset Contract

All datasets are represented by `DatasetBundle`.

| Field | Contract |
|---|---|
| `features` | finite `float64`, shape `(samples, 2)` |
| `targets` | binary `float64`, shape `(samples, 1)` |
| `name` | stable catalog identifier |
| `description` | plain-language data description |
| `purpose` | pedagogical reason for inclusion |
| `seed` | generator seed, zero for fixed truth tables |

The catalog exposes AND, OR, XOR and noisy concentric circles. Truth tables have
four exact samples. Circles are deterministically generated and capped at 1,000
samples so the interface and CPU training remain bounded.
