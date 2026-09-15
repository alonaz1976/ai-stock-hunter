# AI Stock Hunter V5.9.9

V5.9.9 focuses on mobile scan resilience, clearer decision-state UX, and live-trigger trade structure.

## Resilient scanner
- Scan runs in a server-side worker thread and is checkpointed to server disk.
- Switching apps, locking the phone, browser disconnect/reconnect, or Streamlit rerun no longer intentionally owns the scan lifecycle; on return the UI reconnects to the active job while the same server process remains alive.
- Progress shows stage, ticker, percent, elapsed time and dynamic ETA.
- Last completed scan is recoverable from the server checkpoint while that Streamlit instance/storage remains available.
- Successful and skipped counts are shown; skipped tickers and reasons can be expanded.
- Important hosting limitation: a full Streamlit server/container restart can terminate an in-memory worker. Local checkpoint files are not a durable external job queue.

## Scanner UX
- Show: ALL / TOP OPPORTUNITIES / ACTIONABLE NOW / WATCHLIST.
- TOP cards separate status color from numeric TOP/Opportunity values.
- No redundant `PRE-MARKET • PM`; e.g. `PRE-MARKET: WEAKENED (-0.84%)`.
- Green = actual confirmed/live state; yellow = armed/watch/setup/previous-session/pre-open; red = weakened/negative.
- A stage chronology legend appears after TOP cards: WAIT/COLD → WATCH/BUILDING → ARMED → TRIGGER/TRIGGERED → LIVE TRIGGERED, with plain-language explanation and explicit previous-session vs live distinction.

## LIVE TRIGGER levels
Only a genuine `LIVE TRIGGERED` during the regular OPEN session shows model/research trade levels on the TOP card:
- Entry Zone
- Stop / Invalidation
- Target 1 and Target 2
- Risk/Reward to each target
Levels use the existing VWAP/EMA/support/resistance/ATR entry structure. Targets in V5.9.9 are ATR-structured rather than fixed +3%/+6%. They are research levels, not guarantees or orders.

## Sessions retained
- NASDAQ: PRE-MARKET → OPEN → AFTER-MARKET → CLOSED, with real yfinance PM/AH snapshots when available.
- Hong Kong / Tel Aviv: PRE-OPEN → OPEN → CLOSED; no fabricated US-style extended-hours confirmation.
- PM/AH missing data remains N/A rather than being invented.

## Universe / research
- 50 NASDAQ + 50 Hong Kong + 50 Tel Aviv (150 total).
- Hong Kong includes 1196.HK (Realord) and 1570.HK.
- Existing Deep Analysis, hourly/15m confirmation, Entry, Explosive/Timing, walk-forward/backtest and Excel export layers are retained.


## V5.9.9 deployment hotfix
- Pin Streamlit 1.63.0 for reproducible Community Cloud builds.
- Pin `pyarrow<25` to avoid the known PyArrow 25 Community Cloud build/runtime issue visible in deployment logs.

## Startup-safe hotfix C
- Restores the dependency list used by the previously working V5.9.8 (no forced PyArrow pin).
- Quant engine import is caught and displayed inside the app instead of producing a generic pre-render crash.
- Background-scan job storage is initialized lazily, avoiding filesystem side effects before Streamlit renders.
- Build: V599-STARTUP-SAFE-20260915-C.
