# AI Stock Hunter V5.8.2 — Unique Lift / Ablation

V5.8.2 keeps the V5.8.1 validation, overlap/correlation, chart periods, Entry Validation, Scanner, 4% target option and Validation 50 workflow, and adds a unique-contribution gate before any live weight changes.

## What changed
- Added **Unique Lift / Ablation** to Walk-Forward Validation.
- Tests Early RVOL, Volume acceleration, Liquidity and ADX trend strength only on accepted unseen chronological validation blocks.
- For each highlighted component, compares forward target hit rate when that component is active versus inactive while at least one of the other highlighted signals is already active.
- Reports active/control rows, stock coverage, stock/fold coverage, hit rates, Unique Lift, incremental hit-rate percentage points and a conservative verdict.
- Outcome is whether the selected validation target is reached by close within the next 1–3 trading days.
- No live scoring weights are changed in V5.8.2.

## Interpretation
Correlation answers whether signals overlap. Unique Lift asks whether a signal still contributes useful predictive information after other highlighted signals are already present. This is an ablation-style diagnostic, not causal proof.

## Files
- `app.py` — Streamlit UI and validation labs
- `quant_engine.py` — features, scoring, calibration, scanner and backtests
- `requirements.txt` — dependencies
- `README.md` — this file

## Build integrity
- App version: `5.8.2`
- Engine version: `5.8.2`
- Build ID: `V582-UNIQUE-LIFT-20260914-A`
- ZIP contains exactly these four files at root level.

Research prototype; not a guarantee or personalized investment recommendation.
