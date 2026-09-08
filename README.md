# AI Stock Hunter V1

A mobile-friendly Streamlit dashboard for the first quantitative engine.

## Included
- RSI, MACD, EMA20/50/200, ATR, OBV, CMF, Relative Volume
- Smart Money Score (0-100)
- Entry Score (0-100)
- Market-behavior Stock Score (0-100)
- Early Accumulation detection
- Entry zone, invalidation, Target 1/2
- Walk-forward backtest (no future data used to create each signal)

## Run
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. `pip install -r requirements.txt`
4. `streamlit run app.py`
5. Upload a daily OHLCV CSV with at least 210 rows.

Required CSV columns: Open, High, Low, Close, Volume (case-insensitive).

## Next milestone
Connect a market-data API, add valuation/fundamentals, then scan NASDAQ + NYSE and rank the strongest candidates.
