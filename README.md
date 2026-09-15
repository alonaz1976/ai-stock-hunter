# AI Stock Hunter V5.9.6

## Main additions
- Keeps the 150-stock universe: 50 NASDAQ + 50 Hong Kong + 50 Tel Aviv, including 1196.HK Realord and 1570.HK.
- Unified TOP Score combines Opportunity Score, Dynamic Prediction, Entry Score and Reliability for scanner ranking.
- Fixes contradictory AVOID labels: Signal is now reconciled with Opportunity Stage and evidence.
- Separates setup state from execution state with CLOSED / PRE-MARKET / OPEN market phase labels.
- TRIGGER is displayed as LIVE TRIGGERED only during the market's regular open session; otherwise it is PRE-MARKET CONFIRM or PREVIOUS SESSION TRIGGER.
- Exports TOP Score, market phase, live stage, previous-session trigger, Global Rank and Market Rank.
- Retains 150-stock filters, mobile chart controls, chart-last Analyze layout and Excel downloads.

## Important
Market phase uses normal weekday trading clocks and does not model exchange holidays or half-days. PRE-MARKET CONFIRM is an operational state label, not a claim that exchange pre-market price/volume has independently confirmed the setup. Opportunity/TOP scores are research decision layers, not probabilities or guarantees.
