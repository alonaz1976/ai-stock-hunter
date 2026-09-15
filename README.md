# AI Stock Hunter V5.9.8

V5.9.8 adds a real NASDAQ after-market execution layer while preserving the 150-stock multi-market scanner and V5.9.7 pre-market logic.

## Market session model
- NASDAQ: PRE-MARKET (04:00–09:30 ET) → OPEN (09:30–16:00 ET) → AFTER-MARKET (16:00–20:00 ET) → CLOSED.
- Hong Kong / Tel Aviv: PRE-OPEN → OPEN → CLOSED. No fabricated US-style extended-hours confirmation is applied.
- Session clocks are operational weekday clocks and do not model exchange holidays or half-days.

## Real NASDAQ extended-hours confirmation
- Pre-market: PMPrice, PMChangePct, PMVolume, PMVolumeStrength, PMConfirmation.
- After-market: AHPrice, AHChangePct, AHVolume, AHVolumeStrength, AHConfirmation.
- PM/AH confirmation states: CONFIRMED, NEUTRAL, WEAKENED, STRONGLY WEAKENED.
- Extended-hours confirmation can re-rank TOP Score; strong weakening reduces the score.
- ACTIONABLE NOW includes confirmed NASDAQ pre-market/after-market candidates and appropriate regular-session candidates.

## Session clarity
Examples:
- `PRE-MARKET • PM: WEAKENED • PREVIOUS SESSION: TRIGGER`
- `OPEN • LIVE TRIGGERED`
- `AFTER-MARKET • AH: CONFIRMED • REGULAR SESSION: TRIGGER`
- `CLOSED • PREVIOUS SESSION: WATCH`
- `PRE-OPEN • PREVIOUS SESSION: ARMED`

Green is reserved for actual confirmation/live action states. Closed, previous-session, pre-open, setup, neutral, armed and watch states are warning/yellow; weakened states are red.

## Scanner
- 50 NASDAQ + 50 Hong Kong + 50 Tel Aviv.
- Hong Kong universe retains `1196.HK` (Realord) and `1570.HK`.
- Show filter: ALL / TOP OPPORTUNITIES / ACTIONABLE NOW / WATCHLIST.
- Excel export includes PM and AH fields, TOP Score, session status, Entry Score and research metrics.

## Important
Yahoo/yfinance extended-hours data can occasionally be delayed, missing, or incomplete. When no real extended-hours snapshot is available the app leaves confirmation as N/A rather than inventing a signal. TOP/Opportunity scores are research decision layers, not probabilities or guarantees.
