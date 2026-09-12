# AI Stock Hunter V5.3.3 FULL

V5.3.3 keeps the V5.3 scoring and calibration logic intact, while improving confidence transparency and chart control.

## What changed
- Confidence weighting changed from 65% Hit Rate / 35% sample factor to **80% Backtest Performance / 20% Sample Reliability**.
- Confidence now shows three separate values in Analyze:
  - Backtest Performance
  - Sample Reliability
  - Combined Confidence
- The formula is shown directly under the metrics for transparency.
- HIGH confidence still requires at least 30 signals and at least 60% historical hit rate; MEDIUM still requires at least 12 signals and at least 45% hit rate.
- Added display-only chart controls:
  - Timeframe: 15m / 1H / 1D / 1W
  - Period: 1M / 3M / 6M / 1Y
- Chart controls **do not change** Quant, Early Prediction, Entry, Calibration, or Backtest calculations.
- 15-minute chart history is capped to 1 month when a longer visual period is selected, to respect the data provider's intraday history limit.
- V5.3 Calibration Lab, tiny-sample backtest protection, scanner, and event study remain intact.

## Model calculations unchanged in V5.3.3
- Quant final score: Daily 78% + Hourly 22% when hourly data is available.
- Early Prediction final score: Daily 70% + Hourly 30% when hourly data is available.
- Precision Entry: uses 15m data when available, otherwise 1H, otherwise daily.
- Calibration Lab remains based on daily historical D-3 / D-2 / D-1 event diagnostics.

## Files
- app.py
- quant_engine.py
- requirements.txt
- README.md


## V5.3.3 hotfix
- Fixes the confidence display mismatch seen when `Backtest Performance` and `Sample Reliability` showed 0.0% while `Combined Confidence` was non-zero.
- The app now recomputes all three confidence values from the same raw `hit_rate` and `Signals` values.
- Small samples still show `hits/sample` in the main Backtest card, but `Backtest Performance` now correctly shows the mathematical hit rate (for example 2/5 = 40.0%).
- Added an app/engine version-sync warning. If `app.py` and `quant_engine.py` are not from the same V5.3.3 package, the app warns before results are trusted.
- Quant, Early Prediction, Entry, Calibration and Backtest signal-generation logic are unchanged from V5.3.1.


## V5.3.3 hotfix
- Engine version tag is explicit and exported through `get_engine_version()`.
- App checks API compatibility before checking the version tag.
- Matching files show `App and quant engine synced: V5.3.3`.
- ZIP contains the four project files at the archive root for easier GitHub upload.
