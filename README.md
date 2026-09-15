# AI Stock Hunter V5.9.1

V5.9.1 is a usability patch on top of V5.9.0. It keeps the quantitative logic unchanged and adds persistent Excel export to every application tab.

## Excel export coverage
- Analyze: summary, scores, Entry, Explosive Move, components, calibration, static-vs-dynamic backtest, event study when available, and daily feature history.
- Scanner: full ranked scan, Explosive Move subset, Hong Kong subset.
- Backtest: threshold comparison table.
- Validate: Early/Quant walk-forward summaries, raw fold rows, stock audit, skipped stocks and overlap table when available.
- Entry Validation: bucket summary, production gate, raw observations, audit and skipped stocks.
- Explosive Lab: complete explosive walk-forward results.

All four files must be uploaded together to GitHub: app.py, quant_engine.py, requirements.txt and README.md.
