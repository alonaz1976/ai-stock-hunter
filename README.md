# AI Stock Hunter V5.9.9 — Cached Worker Fix

This build keeps the V5.9.9 features but replaces the previous filesystem/builtins/pickle background-job registry with a Streamlit cached server resource.

## Included
- 150-stock scanner and market filters
- NASDAQ real pre-market and after-market layers
- ALL / TOP OPPORTUNITIES / ACTIONABLE NOW / WATCHLIST
- server-side scan worker that survives browser disconnects while the Streamlit server process remains alive
- scan progress, elapsed time and dynamic ETA
- Last Completed Scan in server memory
- skipped-stock reasons
- simplified session wording and corrected status-color logic
- setup chronology legend
- LIVE TRIGGER entry zone / invalidation / targets / risk-reward

## Reliability change
No scan-job directory, pickle file, JSON checkpoint, or builtins registry is touched at application startup. Worker threads do not call Streamlit APIs. This avoids the startup/redeploy path introduced in the first V5.9.9 builds.

A full Streamlit server/container restart clears in-memory scan jobs; browser disconnect/reconnect does not.
