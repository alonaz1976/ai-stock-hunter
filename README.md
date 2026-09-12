# AI Stock Hunter V5.3 FULL

V5.3 focuses on evidence quality and Early Prediction calibration for a single-stock analysis.

## What changed
- Early Prediction Calibration Lab for every strong single-day move defined by the selected Target %.
- Component-level diagnostics for both Early and Quant: event coverage, pre-event activation, baseline activation, lift, average strength, and D-3/D-2/D-1 activation.
- Tiny backtest samples are no longer highlighted as a seemingly reliable percentage. Below 12 signals the UI shows hits/sample and LOW SAMPLE.
- Threshold Backtest Lab hides Hit Rate % below 12 signals and reports Hits + Sample Quality instead.
- Existing scanner, entry engine, event study and scoring formulas remain intact.

## Files
- app.py
- quant_engine.py
- requirements.txt
- README.md

## Important interpretation
Lift > 1 means a component was active more often before selected strong up-days than during the historical baseline. This is diagnostic evidence, not proof of causality and not a buy signal. A useful component should ideally combine lift above 1 with meaningful event coverage and a sufficiently large event sample.
