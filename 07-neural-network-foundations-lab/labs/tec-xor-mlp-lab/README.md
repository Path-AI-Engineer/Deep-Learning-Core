# XOR MLP Lab

XOR is not linearly separable: neither a single line nor a single logistic
neuron can assign both diagonal pairs correctly. The approved experiment uses:

- two input features;
- four tanh hidden units;
- one sigmoid output;
- Xavier initialization;
- Binary Cross-Entropy;
- full-batch SGD;
- a fixed seed.

The closure criterion is `100%` training accuracy with a reproducible
configuration and verified gradients. This is a pedagogical fit, not evidence
of generalization from four truth-table samples.
