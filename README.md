# AI Stock Hunter — Quant V4

V4 upgrades the V3 scanner with a modern dashboard, Early Prediction, precision Entry Timing, richer walk-forward backtesting, sample-size-aware confidence, and a threshold comparison lab.

## Install / run
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## V4 highlights
- Quant Score + BUY / WATCH / AVOID
- Early Prediction: OBV, CMF, Accumulation/Distribution, Bollinger/Keltner squeeze, Relative Strength proxy, ROC acceleration, price/volume divergence and early RVOL
- Entry Timing Engine using intraday 15-minute data when available, with 1-hour fallback
- Entry zone, breakout trigger, invalidation, +3% and +6% reference targets
- Candlestick chart with EMA20 / EMA50 and volume
- Backtest: Hit Rate, Sample Size, Average Forward Return, Max Drawdown and Confidence
- Threshold Lab compares 60–75 rather than assuming one BUY threshold is always best

## Important
Quant Score is a setup-quality score, not a probability of profit. Hit Rate is historical and must be interpreted together with Sample Size and the backtest assumptions. Yahoo/yfinance intraday history availability can vary by ticker and exchange.

## V4.1 — Two-stage Broad Scan
- Built-in **Broad 200** universe (200 tickers) plus Core 26 and Custom modes.
- Stage 1 runs a fast daily Quant + Early Accumulation pass across the whole universe.
- Stage 2 keeps the strongest 30 by default (adjustable 10–50) and only then downloads 1h + 15m data.
- Entry Timing, full backtest, Hit Rate / Sample Size / Confidence are calculated for finalists only.
- Ranking now includes Entry Score, Entry Status, Entry Zone, Trigger and Invalidation.
- This architecture is designed to reduce unnecessary intraday downloads; actual scan time still depends on Yahoo Finance/network response and can vary.

## V4.1.1 — Stability fix
- Fixed Bollinger/Keltner `squeeze_release` calculation (`DataFrame.squeeze` name collision).
- Fixed Entry Timing `invalidation` variable name.
- Core feature, backtest, entry-timing and two-stage scan paths were smoke-tested on synthetic OHLCV data.
