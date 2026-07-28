# CNN training lab

Train the MLP baseline and CNN with the same split and protocol. Observe the
explicit transition between `train()` and `eval()`, CrossEntropyLoss on logits,
Adam updates and validation-based checkpoint selection. The test set is not a
hyperparameter-selection tool.
