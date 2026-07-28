# Sequence Contract

The models implement many-to-one classification. Input is batch-first `[batch, time, features]`; the standard window is `[batch, 128, 9]`. A classifier returns logits `[batch, 6]`.

For variable-length batches, lengths are positive integers no greater than the padded time dimension. The recurrent encoder uses `pack_padded_sequence(..., batch_first=True, enforce_sorted=False)` and classifies the last layer's final hidden state. LSTM cell state is internal and never substituted for the hidden representation.

Permuting timesteps is an explicit ablation. It must not be used as data augmentation because it destroys the temporal hypothesis under evaluation.

