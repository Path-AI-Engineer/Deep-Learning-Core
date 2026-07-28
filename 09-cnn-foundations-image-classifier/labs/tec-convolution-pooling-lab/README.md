# Convolution and pooling lab

Use `cross_correlate_2d` with 3 × 3 kernels, stride 1/2 and padding 0/1. Compare
every result against `torch.nn.functional.conv2d`, calculate output shapes and
trace the receptive field through both convolution and max-pooling blocks.
