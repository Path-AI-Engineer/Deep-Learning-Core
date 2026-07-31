# Data contract

The official dataset target is `torchvision.datasets.FashionMNIST`.

| Property | Contract |
|---|---|
| Input shape | `[N, 1, 28, 28]` |
| Data type | `float32` |
| Pixel range | `[0, 1]` |
| Classes | 10 official FashionMNIST classes |
| Train/validation | Deterministic stratified 90/10 split of official training data |
| Test | Untouched official test partition |
| Seed | 42 |
| Labels in AE loss | Never |

Checksums and a split manifest are produced during preparation. Labels may be used
after autoencoder training for stratification, visual colouring and linear-probe
evaluation.

The committed `data/samples` gallery is a deterministic, programmatically generated
fixture. Its shapes resemble clothing silhouettes only to exercise the product. It
is not a substitute for FashionMNIST and is identified as `educational_fixture` by
the API and interface.
