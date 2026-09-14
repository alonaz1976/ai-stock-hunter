# AI Stock Hunter V5.8.1 — Component Overlap & Chart Period Upgrade

V5.8.1 keeps the V5.8 Entry Validation baseline and adds validation safeguards before any live weight changes.

## What changed
- Added chart periods: **1D, 1W, 5D, 1M, 3M, 6M, 1Y, MAX**.
- 1D display automatically uses 5-minute candles; short-week views use intraday candles when needed.
- Added **Component overlap / correlation** research to Walk-Forward Validation.
- Correlation uses normalized component strength and Spearman correlation across validated daily rows.
- Highlights Early RVOL, Volume acceleration, Liquidity and ADX trend strength.
- Shows the highest component overlaps and warns when strong components have correlation >= 0.70, reducing double-counting risk before reweighting.
- Keeps the V5.8.0 Entry Validation, Scanner progress, 4% target, Validation 50 universe and Static-vs-Dynamic baseline unchanged.

## Important methodology note
Correlation is a redundancy warning, not proof that a component should be removed. Weight changes should wait for multi-target / multi-history validation and unique-contribution testing.

## Files
- `app.py` — Streamlit UI and validation labs
- `quant_engine.py` — features, scoring, calibration, scanner and backtests
- `requirements.txt` — dependencies
- `README.md` — this file

## Build integrity
- App version: `5.8.1`
- Engine version: `5.8.1`
- Build ID: `V581-CORR-CHART-20260914-A`
- ZIP must contain exactly these four files at root level.

This is a research prototype, not a guarantee or personalized investment recommendation.
