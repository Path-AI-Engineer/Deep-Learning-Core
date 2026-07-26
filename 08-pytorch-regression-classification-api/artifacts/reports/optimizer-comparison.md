# Optimizer Comparison

The bounded Wine experiment keeps the split, architecture, batch size and seed fixed.
Adam is the approved default because it reaches a lower validation loss within the fixed
budget. SGD remains available for the learning lab, not for automated search.

Dropout is limited to 0.10 and weight decay to 0.0001. These choices reduce the
train/validation gap without claiming global optimality.
