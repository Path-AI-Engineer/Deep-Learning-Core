# Data Contract

Every dataset uses a deterministic 70/15/15 train/validation/test split. Classification
splits are stratified. The test split is isolated until final evaluation.

California Housing uses eight numeric features in the official scikit-learn order. The target
`MedHouseVal` is expressed in USD 100,000. Wine uses thirteen numeric measurements and three
stable class labels.

`StandardScaler` is fitted only on the training feature matrix. Regression targets remain in
their original unit. Missing, non-finite or reordered input values are rejected.

## California Housing source selection

The loader uses `data/raw/california_housing.csv` when the complete official
dataset has been prepared locally. If that file is absent, the repository's
149-row official-source reference sample supports deterministic offline tests
and product demonstrations. Only when neither file exists does the loader call
scikit-learn's fetcher. The selected source and sample count are embedded in
the model bundle so a reference-sample artifact cannot be mistaken for a
full-dataset benchmark.
