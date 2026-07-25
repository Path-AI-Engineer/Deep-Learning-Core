# Single Neuron Lab

Use fixed values to inspect one affine transform:

```text
x = [1, 3]
w = [0.5, 2.0]
b = 0.1
z = 1(0.5) + 3(2.0) + 0.1 = 6.6
```

`tests/unit/test_dense.py` protects this exact calculation. The dense layer then
generalizes the same operation across a batch and multiple neurons.
