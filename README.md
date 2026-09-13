# AI Stock Hunter V5.4.0 — Dynamic Calibration

V5.4.0 adds ticker-specific dynamic calibration while keeping the prior static model visible as a benchmark.

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
