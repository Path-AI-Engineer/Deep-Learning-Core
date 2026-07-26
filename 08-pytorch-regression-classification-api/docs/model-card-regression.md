# Model Card — California Housing Regression MLP

## Intended use

Educational estimation of median California district house value in units of USD 100,000.

## Evaluation

The approved checkpoint is selected on validation loss and must beat a training-mean
`DummyRegressor` on test MAE. MAE, RMSE and R² are exported in the active bundle.

## Limitations

The source dataset reflects historic district aggregates and is not a current property
valuation service. Outputs are estimates and must not support financial decisions.

## Verified release evidence

- Model version: `v1.0.0`
- Test MAE: `0.5097`
- Test RMSE: `0.8002`
- Test R²: `-0.0505`
- Acceptance: model MAE is lower than the mean-regressor baseline MAE.

The checked-in bundle was produced from the 149-row official-source reference
sample because the build environment had no dataset network access. The
negative R² and the small reference sample make this artifact appropriate for
workflow demonstration, not a full benchmark claim. The reproducible training
path accepts the complete official CSV at
`data/raw/california_housing.csv`.
