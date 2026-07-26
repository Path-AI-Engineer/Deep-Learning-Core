# Train and evaluation mode lab

Dropout is stochastic under `model.train()` and disabled under `model.eval()`. The lab proves
that evaluation is deterministic and does not build a gradient graph.
