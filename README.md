# AI Stock Hunter V5.9.9 STABLE

Startup-stable build after diagnostic isolation on Streamlit Cloud.

- Keeps V5.9.8 real NASDAQ pre-market + after-market confirmation.
- Keeps unified TOP Score, smart Show filter, session-aware colors/status wording.
- Adds scan progress with elapsed time + dynamic ETA.
- Adds skipped-stock counts/reasons.
- Adds setup chronology legend after TOP cards.
- Adds Entry Zone / Stop / Target 1 / Target 2 / Risk-Reward on genuine LIVE TRIGGERED states.
- Removes the V5.9.9 cached background-worker startup path that was isolated as the new startup difference versus working V5.9.8.

Important: this stable build runs the scan synchronously. If the mobile browser is suspended/disconnected during a scan, continuation is not guaranteed. Background persistence will be reintroduced only with a deployment-safe architecture.
