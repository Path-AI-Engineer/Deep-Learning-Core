# Sequence Shapes Lab

Verify the transformations `[N,128,9] -> batches -> [batch,6]`, channel order, label range, and grouped subject split. Success means no subject leakage and normalization statistics fitted only on training data.

Run `python scripts/prepare_data.py`, then inspect `data/processed/preparation_manifest.json`.

