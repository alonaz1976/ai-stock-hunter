# AI Stock Hunter V5.4.6 — Dynamic Calibration

V5.4.6 adds ticker-specific dynamic calibration while keeping the prior static model visible as a benchmark.

## What changed
- Dynamic Early and Dynamic Quant weights use historical Lift, Coverage, sample reliability and D-3/D-2/D-1 stability.
- Weight changes are shrunk toward the base model for small samples and capped to reduce overfitting.
- Final Prediction dynamically combines calibrated Quant and Early scores within bounded weights.
- Static vs Dynamic holdout backtest: calibration is fit on the earlier history and evaluated on later unseen rows.
- Backtest adds Baseline Hit Rate and Signal Lift.
- Scanner ranks by Dynamic Prediction and shows Dynamic Early, Dynamic Quant, Signal Lift and Calibration Confidence.
- Cross-family Early + Quant combination research table is included in Analyze.
- Static scores remain visible for comparison during validation.

## Files
- `app.py` — Streamlit UI
- `quant_engine.py` — features, scoring, dynamic calibration, scanner and backtests
- `requirements.txt` — dependencies
- `README.md` — this file

## Important
This is a research prototype, not a guarantee or personalized investment recommendation. Dynamic calibration reduces fixed-weight bias but can still overfit, especially on small event samples. Validate across multiple tickers and longer histories before changing the base model permanently.


## V5.4.6 hotfix
- Prevents a hard ImportError when app.py and quant_engine.py are temporarily out of sync during deployment.
- Adds explicit engine API validation for the dynamic calibration functions.
- Keeps the four-file root deployment package.


## Build integrity
- Engine build: `V543-DYNAMIC-20260913-A`
- Required dynamic API: `dynamic_scores`, `compare_static_dynamic_backtest`, `scan_universe_dynamic`.
- The ZIP must contain exactly four files at root level.


V5.4.6 robustness fix: the dynamic-calibration layer has a verified in-app fallback, so a stale Streamlit engine cache cannot block Analyze, Scanner, or Dynamic Backtest.


## V5.5.0 — Multi-Stock Validation
Adds a Validate tab that runs the existing ticker calibration across a diversified basket and aggregates Median Lift, Mean Lift, Coverage, positive-stock consistency, stability, and total event count. Research-only: it does not automatically alter live model weights.


## V5.7.1 — Out-of-Sample Validation Basket
- Expands the default validation basket to 50 diversified stocks.
- Splits each stock chronologically: first 70% discovery, final 30% unseen validation.
- Reports Discovery Median Lift vs Validation Median/Mean Lift, coverage, positive-stock consistency, stability and validation event count.
- Classifies components as OOS WINNER, PROMISING, MIXED / WEAK or FAILED OOS.
- Every skipped stock now has an explicit reason.
- Does not alter live scoring weights; this is a validation gate before any future reweighting.


## V5.7.1
- Fixed Liquidity using relative dollar turnover vs rolling 60-bar median.
- min_turnover remains a separate tradability floor.
- Replaced one 70/30 split with four expanding chronological walk-forward folds.
- Added cross-stock + cross-fold robustness metrics.
- Live weights remain unchanged pending V5.8 optimization.


## V5.7.1
- Added the exact Validation 50 basket as a one-click Scanner universe.
- Validation 50 defaults to deep-analysis of all 50 names.
- Scanner results are explicitly ranked best to worst by Dynamic Prediction, with Dynamic Quant, Dynamic Early and Entry Score as tie-breakers.
- Added Rank to cards and result table.
- Validation and Scanner now share one `VALIDATION_50` source list to prevent drift.
- Fixed app/engine version and build identifiers so sync checks are truthful.


## V5.8.0 — Entry Validation + Stable Chart
- Added 4% as a selectable target across Analyze, Scanner, Backtest, Walk-Forward Validation and Entry Validation.
- Scanner now shows a real two-stage progress bar: daily pre-scan and deep intraday analysis.
- Added a dedicated Entry Validation tab using the exact shared Validation 50 basket.
- Entry Validation compares Static vs Dynamic Entry Score on four chronological walk-forward folds, with score buckets, hit rate, lift, forward return, drawdown, time-to-target and a production-like Entry ≥72 + Quant ≥66 gate.
- Entry Validation preserves the current live formula; Dynamic differs through Dynamic Quant. The current production Entry formula accepts Early Score but does not yet use it.
- Validation history options now include 1Y, 2Y, 3Y and 5Y.
- Rebuilt the chart interaction layer with persistent UI state, 5D/1M/3M/6M/1Y/MAX range buttons, double-click reset, mobile-safe zoom behavior, selectable EMA9/EMA20/EMA50/VWAP/Volume layers, and optional RSI/MACD panels.
- App and engine build: `V580-ENTRY-CHART-20260914-A`.
- Package remains exactly four root files.
