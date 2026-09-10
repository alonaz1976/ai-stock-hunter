# AI Stock Hunter — Quant V4.1.6 Stable Data Layer

This build replaces the fragile yfinance column handling with a strict OHLCV normalization layer.

Key fixes:
- Rebuilds every market-data response into exactly Open / High / Low / Close / Volume.
- Supports flat columns, duplicate labels, and yfinance MultiIndex columns in either level order.
- Converts all OHLCV inputs to float64 before indicators are calculated.
- Removes boolean duplicate-column masks from compute_features.
- Isolates failures per ticker in the multi-stock scanner, so one malformed symbol does not crash the full scan.
- Keeps Quant V4 features: Early Prediction, 1h/15m Entry Timing, Confidence, Walk-forward Backtest, Broad 200 two-stage scan.

Run:
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Research prototype only. Quant scores, backtests and entry zones are estimates, not guarantees.
