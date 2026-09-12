# AI Stock Hunter V5.2.2 FULL

Hotfix release.

Included files:
- app.py
- quant_engine.py
- requirements.txt
- README.md

Changes in V5.2.2:
- Fixes Streamlit ImportError caused by the new Early Event Backtest import.
- Early Event Backtest is now self-contained in app.py and uses the existing Quant/Early scoring functions.
- Keeps the single-stock event analysis based on the selected Target %.
- Keeps the mobile tab layout fix.


## V5.2.2 hotfix
Fixed the Analyze > Why this stock diagnostic KeyError. Feature-only fields (squeeze, squeeze_release, pv_divergence) are now read from the latest computed feature row. This allows the Early Prediction event-study table to render after the chart.
