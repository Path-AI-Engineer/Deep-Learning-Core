# Image tensor lab

Trace one FashionMNIST image from PIL grayscale pixels to a normalized float32
NCHW tensor. Verify shape `[1, 1, 28, 28]`, value scaling, mean/std
normalization and inverse preview reconstruction. Validation and test transforms
must remain deterministic.
