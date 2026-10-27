# Metrics and model expansion

## Background and context

The user approved all metrics and algorithms in the preceding gap analysis. The
workbench currently has 84 runnable models, hard-coded search objectives, and a
single tabular holdout split. Existing uncommitted user work must be preserved.

## Acceptance criteria

- Classification, regression, labeled clustering/anomaly, and forecasting metrics
  are computed from genuine predictions/scores; undefined metrics are null with a
  reason. AP and trapezoidal PR area have distinct names.
- Search supports task-compatible objectives and derives their direction.
- New models run through the existing lifecycle: validation, explicit freeze,
  final refit/test, export, reload, and genuine supported inference.
- Add Dummy, Voting/Stacking/Bagging, Gaussian mixtures, HDBSCAN, Bayesian/ARD/
  kernel/robust/PLS regression, UMAP, naive/seasonal-naive/exponential-smoothing/
  ARIMA/SARIMA forecasting, CNN/TCN/Transformer window regression, and autoencoder
  dimensionality reduction.
- Provide optional fold evaluation, group splits, and chronological/rolling
  evaluation without consuming holdout test rows or fitting preprocessing outside
  fold training data. Every real search fit consumes its budget.
- GUI and advanced batch configuration expose the additions; results and settings
  remain serializable and older default configuration hashes remain stable.
- Run the existing tests and synthetic examples, new differential/integration
  tests, optional-model smoke tests, and an offscreen GUI check.

## Implementation details

1. Add a shared metric registry and evaluators, integrate score caches and deep
   classification outputs, and expand objective resolution.
2. Add JSON-configurable ensemble adapters and traditional estimators to the
   catalog. Add optional extras for UMAP and statsmodels; keep imports lazy.
3. Add chronological forecasting adapters and window neural architectures, plus a
   portable autoencoder transformer, reusing existing freeze/export contracts.
4. Extend split settings and add fold evaluation with fold-local pipelines and
   fit accounting. Keep the final test partition reserved and unchanged.
5. Expose evaluation/metrics through the GUI and batch requests, preserve settings,
   update model counts and guides, and record actual verification in version.md.

## Dependencies and validation strategy

Use the locked project Python and installed scikit-learn/PyTorch/skorch. Resolve
new optional dependency versions against the package index and refresh uv.lock.
Compare metric results against scikit-learn on seeded binary/multiclass/skewed and
boundary cases. Test missing probability support, invalid domains, class order,
group/time isolation, fold-local preprocessing, budgets, recovery, persistence,
and once-only final testing. Forecasts are fixed-origin predictions with explicit
lead times; MASE denominators come only from training targets. Do not claim an
updated Windows binary unless one is actually built and verified.
