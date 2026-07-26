# Model Card — Wine Classification MLP

## Intended use

Educational multiclass inference over the thirteen numeric Wine measurements.

## Evaluation

The approved model is selected on validation loss and must exceed a prior-only
`DummyClassifier` on test macro-F1. Accuracy, macro-F1, log loss and confusion matrix are
exported in the active bundle.

## Limitations

The dataset is small and curated. Probabilities are not certainty, quality assessments or
causal conclusions. Inputs outside observed feature ranges may be unreliable.

## Verified release evidence

- Model version: `v1.0.0`
- Test macro F1: `0.9599`
- Test accuracy: `0.9630`
- Test log loss: `0.0840`
- Confusion matrix: `[[9, 0, 0], [0, 11, 0], [0, 1, 6]]`
- Acceptance: macro F1 exceeds the prior-only dummy classifier.
