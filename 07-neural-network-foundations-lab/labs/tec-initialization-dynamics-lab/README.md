# Initialization Dynamics Lab

The engine supports four strategies:

| Strategy | Educational observation |
|---|---|
| Zeros | Hidden units begin symmetrically and receive identical signals |
| Small normal | Breaks symmetry but may shrink signals in deeper networks |
| Xavier | Scales variance for sigmoid/tanh-style activations |
| He | Scales variance for ReLU-style activations |

The lab compares initial predictions, loss and gradient norms under a fixed
dataset and seed. These observations explain initialization behavior for this
small MLP; they do not replace architecture-specific empirical validation.
