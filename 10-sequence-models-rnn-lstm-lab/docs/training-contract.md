# Training Contract

Training is deterministic on CPU with seed 42. The optimizer is AdamW, the loss is multiclass cross-entropy, and model selection uses validation macro F1.

Every epoch records training loss, validation loss, accuracy, macro F1, and gradient norms before and after clipping. Early stopping restores the highest-scoring validation state. The official test split is evaluated only after selection.

Changing a dataset checksum, split, preprocessing statistic, architecture, or weight file requires a new bundle version. API requests never start training.

