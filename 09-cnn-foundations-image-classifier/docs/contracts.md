# Contracts

## Data

- Dataset: FashionMNIST.
- Input: one grayscale channel, 28 × 28 pixels.
- Stable classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt,
  Sneaker, Bag and Ankle boot.
- Validation is selected deterministically from the official training split.
- The official test set remains isolated until the final evaluation.
- Augmentation is permitted only in the training transform.

## Inference

- Tensor: float32 NCHW `[1, 1, 28, 28]`.
- Output: ten raw logits; probabilities are created with softmax for presentation.
- Uploads: PNG or JPEG, at most 2 MiB, decoded and discarded in memory.
- CPU load uses `state_dict`, `map_location="cpu"`, `weights_only=True` and
  `model.eval()`.

## Explanations

Only `conv1`, `pool1`, `conv2` and `pool2` may be captured. A temporary forward
hook is registered for one request and removed in `finally`. Feature maps are
descriptive activations. They do not prove causality or human-like reasoning.

## HTTP

The public contract is under `/api/v1`. Error responses use an HTTP status and a
non-sensitive `detail`; every response receives `X-Request-ID`.

The cross-correlation lab is model-independent. Model, gallery and evaluation
routes return 503 when the approved assets are absent.
