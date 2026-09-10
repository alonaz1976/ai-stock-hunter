
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from quant_engine import fetch_ohlcv, compute_features, score_latest, backtest_signal, scan_universe

st.set_page_config(
    page_title="AI Stock Hunter — Quant V3",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----- Styling: mimic the user's prior AI Stock Hunter UI -----
st.markdown("""
<style>
:root{
    --bg:#0f1015;
    --panel:#24252d;
    --muted:#9b9ba4;
    --text:#f4f4f7;
    --accent:#ff5b61;
    --line:#2c2d35;
    --good:#4fd1a5;
    --warn:#f1c75b;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--text);
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
.block-container{
    max-width: 760px;
    padding-top: 2.3rem;
    padding-bottom: 5rem;
}
h1,h2,h3,h4,p,label,span,div { font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; }
.hero-title{
    font-size: 4.35rem;
    line-height: 1.03;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin: 3.7rem 0 1.6rem 0;
}
.hero-sub{
    color: var(--muted);
    font-size: 1.33rem;
    line-height: 1.75;
    margin-bottom: 1.5rem;
}
.small-muted{
    color: var(--muted);
    font-size: 1.02rem;
}
.section-title{
    font-size: 2.0rem;
    font-weight: 800;
    margin: 2.2rem 0 1rem 0;
}
.card{
    background: var(--panel);
    border: 1px solid #31323a;
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.signal-buy{color:var(--good);font-weight:800}
.signal-watch{color:var(--warn);font-weight:800}
.signal-no{color:var(--muted);font-weight:700}
.score-big{
    font-size:2.1rem;
    font-weight:800;
}
hr{
    border:none;
    border-top:1px solid var(--line);
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea{
    background: var(--panel)!important;
    border-color: #33343c!important;
    border-radius: 14px!important;
}
.stTextArea textarea{
    background: var(--panel)!important;
    color: var(--text)!important;
    min-height: 150px!important;
}
.stTextInput input{
    background: var(--panel)!important;
    color: var(--text)!important;
}
.stSelectbox label,.stTextInput label,.stTextArea label,.stSlider label,.stNumberInput label{
    color:var(--text)!important;
    font-size:1.05rem!important;
}
.stButton > button{
    width:100%;
    border-radius:14px;
    min-height:58px;
    background:#151923;
    color:white;
    border:1px solid #45464f;
    font-size:1.08rem;
}
.stButton > button:hover{
    border-color:var(--accent);
    color:white;
}
[data-testid="stTabs"] [role="tablist"]{
    gap: 22px;
    border-bottom: 1px solid var(--line);
}
[data-testid="stTabs"] button[role="tab"]{
    font-size:1.08rem;
    padding-left:0;
    padding-right:0;
}
[data-testid="stTabs"] button[aria-selected="true"]{
    color:var(--accent)!important;
    border-bottom:3px solid var(--accent)!important;
}
[data-testid="stMetricValue"]{
    font-size:1.65rem;
}
.stDataFrame{
    border-radius:14px;
    overflow:hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-title">📈 AI Stock<br>Hunter — Quant V3</div>
<div class="hero-sub">Smart Money • Short-Term Quant Signals • Early Accumulation • Multi-stock Scanner</div>
""", unsafe_allow_html=True)

tab_analyze, tab_scan = st.tabs(["🔎 Analyze one stock", "🔥 Scan watchlist"])

# Shared defaults
DEFAULT_TICKERS = "1196.HK,NVDA,NVMI,MU,AMD,AVGO,AMZN,META,GOOGL,MSFT,AAPL,TSLA,PLTR,CRWV,NBIS,GILD,ORCL,SMCI,ARM,TSM,QCOM,NFLX,UBER,COIN,HOOD,TTWO"

def signal_from_score(score, threshold=74):
    if score >= threshold:
        return "BUY CANDIDATE"
    if score >= threshold - 8:
        return "WATCH"
    return "NO SIGNAL"

def signal_class(s):
    return {"BUY CANDIDATE":"signal-buy","WATCH":"signal-watch","NO SIGNAL":"signal-no"}.get(s,"signal-no")

def safe(x, d=2):
    try:
        if pd.isna(x): return "—"
        return f"{float(x):.{d}f}"
    except:
        return "—"

with tab_analyze:
    st.markdown('<div class="section-title">Market data</div>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker", "1196.HK", key="one_ticker")
    history_period = st.selectbox("History period", ["3mo","6mo","1y"], index=1, key="one_period")
    horizon = st.selectbox("Forecast horizon", [3,5,7,10], index=1, key="one_horizon")
    target = st.selectbox("Target move for backtest", [3,5,6,8,10,15,20], index=2, key="one_target")
    use_hourly = st.checkbox("Use hourly confirmation", value=True, key="one_hourly")

    if st.button("Analyze stock", key="analyze_btn"):
        with st.spinner("Analyzing quantitative setup..."):
            d = fetch_ohlcv(ticker.strip().upper(), period=history_period, interval="1d")
            if d is None or len(d) < 35:
                st.error("Not enough market data for this ticker.")
            else:
                f = compute_features(d)
                latest = score_latest(f, min_turnover=0)
                hscore = np.nan
                if use_hourly:
                    h = fetch_ohlcv(ticker.strip().upper(), period="1mo", interval="1h")
                    if h is not None and len(h) >= 30:
                        hf = compute_features(h, intraday=True)
                        hscore = score_latest(hf, min_turnover=0)["score"]
                score = latest["score"] if pd.isna(hscore) else 0.78*latest["score"] + 0.22*hscore
                sig = signal_from_score(score)
                bt = backtest_signal(f, horizon=int(horizon), target_pct=float(target)/100, score_threshold=max(55,score-8), min_turnover=0)

                st.markdown('<div class="section-title">Quant signal</div>', unsafe_allow_html=True)
                c1,c2,c3 = st.columns(3)
                c1.metric("Quant Score", f"{score:.1f}/100")
                c2.metric("Price", safe(latest["price"],3))
                c3.metric("Signal", sig)

                st.markdown(f"""
                <div class="card">
                    <div class="{signal_class(sig)}" style="font-size:1.25rem">{sig}</div>
                    <div class="small-muted" style="margin-top:8px">
                    Short-term quantitative setup for {ticker.strip().upper()} based on momentum, volume acceleration,
                    breakout proximity, RSI, MACD, ADX, ATR and liquidity.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Vol Ratio", safe(latest.get("volume_ratio"),2)+"x")
                c2.metric("RSI", safe(latest.get("rsi14"),1))
                c3.metric("ADX", safe(latest.get("adx14"),1))
                c4.metric("ATR %", safe(latest.get("atr_pct"),2)+"%")

                st.markdown("#### Price")
                st.line_chart(d[["Close"]].tail(70), use_container_width=True)

                st.markdown("#### Volume")
                st.bar_chart(d[["Volume"]].tail(35), use_container_width=True)

                st.markdown("#### Signal breakdown")
                comp = pd.DataFrame(latest["components"], columns=["Component","Points","Max"])
                comp["% of max"] = (comp["Points"]/comp["Max"]*100).round(0)
                st.dataframe(comp, use_container_width=True, hide_index=True)

                st.markdown("#### Walk-forward check")
                if bt["n"] > 0:
                    st.write(f"Historical hit rate: **{bt['hit_rate']*100:.1f}%** "
                             f"({bt['n']} similar historical setups) for reaching **+{target}%** within **{horizon} trading days**.")
                    st.progress(min(max(bt["hit_rate"],0),1))
                else:
                    st.info("Not enough comparable historical setups at this threshold.")

    st.markdown('<div class="small-muted" style="margin-top:26px">Research prototype — quantitative signals are not guarantees or personalized investment advice.</div>', unsafe_allow_html=True)

with tab_scan:
    st.markdown('<div class="section-title">Multi-stock opportunity scanner</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="font-size:1.12rem">Scans a focused liquid watchlist and ranks only the strongest short-term quantitative setups.</div>', unsafe_allow_html=True)

    tickers_text = st.text_area("Tickers to scan (comma separated)", DEFAULT_TICKERS, key="scan_tickers")
    history = st.selectbox("History period", ["3mo","6mo","1y"], index=1, key="scan_period")
    horizon2 = st.selectbox("Forecast horizon", [3,5,7,10], index=1, key="scan_horizon")
    target2 = st.selectbox("Target move for backtest", [3,5,6,8,10,15,20], index=2, key="scan_target")
    min_score = st.slider("BUY threshold", 60, 90, 74, key="scan_threshold")
    only_signals = st.checkbox("Show only BUY / WATCH", value=True, key="only_sig")
    hourly2 = st.checkbox("Use hourly confirmation", value=True, key="scan_hourly")

    if st.button("Scan watchlist", key="scan_btn"):
        tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
        with st.spinner("Scanning watchlist..."):
            res = scan_universe(
                tickers=tickers,
                daily_period=history,
                use_hourly=hourly2,
                hourly_period="1mo",
                horizon=int(horizon2),
                target_pct=float(target2)/100,
                min_turnover=0
            )
        if res.empty:
            st.warning("No valid results returned.")
        else:
            res["Signal"] = res["Score"].apply(lambda s: signal_from_score(s,min_score))
            if only_signals:
                res = res[res["Signal"].isin(["BUY CANDIDATE","WATCH"])]

            st.session_state["last_scan_v3"] = res
            st.session_state["last_scan_time_v3"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "last_scan_v3" in st.session_state:
        res = st.session_state["last_scan_v3"].copy()
        if not res.empty:
            st.markdown('<div class="section-title">Top opportunities</div>', unsafe_allow_html=True)
            st.caption("Last scan: " + st.session_state.get("last_scan_time_v3",""))
            res = res.sort_values("Score", ascending=False)

            top = res.head(5)
            for _, r in top.iterrows():
                sig = r["Signal"]
                st.markdown(f"""
                <div class="card">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <div>
                      <div style="font-size:1.35rem;font-weight:800">{r['Ticker']}</div>
                      <div class="{signal_class(sig)}">{sig}</div>
                    </div>
                    <div style="text-align:right">
                      <div class="score-big">{r['Score']:.1f}</div>
                      <div class="small-muted">Quant Score</div>
                    </div>
                  </div>
                  <hr>
                  <div class="small-muted">
                    Price {safe(r.get('Price'),3)} • Vol Ratio {safe(r.get('VolumeRatio'),2)}x •
                    RSI {safe(r.get('RSI14'),1)} • ADX {safe(r.get('ADX14'),1)} •
                    Hit Rate {safe(r.get('EmpiricalHitRate'),1)}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Full ranking")
            columns = ["Ticker","Signal","Score","Price","DailyChangePct","VolumeRatio","VolumeZ",
                       "RSI14","ADX14","ATRpct","Breakout20Pct","HourlyScore","EmpiricalHitRate","BacktestN"]
            st.dataframe(res[[c for c in columns if c in res.columns]], use_container_width=True, hide_index=True)
        else:
            st.info("No BUY/WATCH signals matched the current threshold.")

    st.markdown('<div class="small-muted" style="margin-top:26px">The scanner can return zero ideas. No signal is a valid result.</div>', unsafe_allow_html=True)
