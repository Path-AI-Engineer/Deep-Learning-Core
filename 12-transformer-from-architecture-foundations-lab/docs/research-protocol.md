# Research protocol

## Questions

1. Can a manually assembled Transformer learn copy, reverse and associative recall?
2. How does performance change when validation sequences exceed trained lengths?
3. What changes with positional signal, head count and LayerNorm placement?
4. How does the Transformer compare with a GRU under the same data contract?

## Registered evidence order

1. Generate deterministic train, validation-ID, validation-OOD and frozen-test splits.
2. Assert canonical-hash disjointness before training.
3. Select configuration using validation evidence only.
4. Repeat the approved comparison over the registered seeds.
5. Freeze code, configuration and hashes.
6. Open the test split once for the final report.

The quick bundle produced by `bootstrap_reference_bundle.py` stops after step 3 and is
marked `reference_validation`.

## Metrics

- exact sequence match;
- token accuracy excluding `PAD`;
- EOS correctness and stop reason;
- metrics by task and content length;
- median and p90 CPU latency;
- training time and optimizer steps.

## Interpretation boundary

Length extrapolation on symbolic tasks is not language generalization. Attention
weights help inspect routing patterns but do not establish causal explanations.
Single-seed success does not establish stability.
