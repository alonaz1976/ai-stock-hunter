# AI Stock Hunter V5.8.3 — Entry Component Validation

V5.8.3 keeps the V5.8.2 Walk-Forward, Unique Lift/Ablation, component-overlap research, stable chart periods, Scanner progress and Entry Validation baseline.

## What changed
- Added **Entry Component Validation** inside the Entry Validation tab.
- Decomposes the live Entry formula into: Price ≥ VWAP, EMA9 ≥ EMA20, Volume Ratio ≥ 1.15, MACD histogram > 0, RSI 50–70, Static Quant ≥ 66 and Dynamic Quant ≥ 66.
- Adds Smart Money research candidates: **Early RVOL, Volume acceleration, Liquidity, OBV accumulation, CMF money flow and Accumulation/Distribution**.
- Tests every component only on unseen chronological walk-forward rows from the same Validation 50 universe.
- Reports active rows, coverage, Hit Rate, Lift vs baseline, inactive Hit Rate, **Unique Lift**, incremental hit-rate points, forward return, drawdown, positive stocks and positive folds.
- Does **not** change Entry, Early or Quant live weights. This is an evidence-gathering release before V5.9.

## Existing V5.8.x safeguards retained
- Targets include 3%, 4%, 5%, 6% and 8%.
- Entry Validation compares Static vs Dynamic on the same 50-stock universe.
- Early/Quant Validation includes component overlap/correlation and Unique Lift/Ablation.
- Chart periods include 1D, 1W, 5D, 1M, 3M, 6M, 1Y and MAX.

## Build integrity
- App version: `5.8.3`
- Engine version: `5.8.3`
- Build ID: `V583-ENTRY-COMP-20260914-A`
- ZIP contains exactly `app.py`, `quant_engine.py`, `requirements.txt`, `README.md` at root level.

This is a research prototype, not a guarantee or personalized investment recommendation.
