
import streamlit as st
import pandas as pd
from quant_engine import (
    fetch_ohlcv, compute_features, score_latest, backtest_signal,
    scan_universe, DEFAULT_WEIGHTS
)

st.set_page_config(page_title="Short-Term Quant Scanner", layout="wide")
st.title("Short‑Term Quant Scanner")
st.caption("סורק כמותי לטווח של כמה ימי מסחר. כלי מחקרי — לא הבטחת תשואה.")

with st.sidebar:
    st.header("הגדרות")
    horizon = st.slider("טווח תחזית (ימי מסחר)", 2, 10, 5)
    target_pct = st.slider("יעד מהלך לבדיקה", 2.0, 20.0, 6.0, 0.5) / 100
    daily_period = st.selectbox("חלון יומי", ["3mo", "6mo", "1y"], index=1)
    use_hourly = st.checkbox("שלב נתוני שעה", value=True)
    hourly_period = st.selectbox("חלון שעתי", ["5d", "1mo", "3mo"], index=1)
    min_score = st.slider("סף איתות", 50, 90, 72)
    min_turnover = st.number_input("מחזור כספי מינימלי ליום (מטבע מקומי)", value=3_000_000, step=500_000)
    st.markdown("---")
    st.caption("למניות הונג קונג: לדוגמה 1196.HK")

default_tickers = "1196.HK, NVDA, MU, NVMI, NBIS, GILD, AMD, AVGO, ORCL, HOOD"
tickers_text = st.text_area("מניות לסריקה — מופרדות בפסיק", default_tickers, height=90)
tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]

uploaded = st.file_uploader("או העלה CSV עם עמודה בשם ticker", type=["csv"])
if uploaded is not None:
    df_up = pd.read_csv(uploaded)
    if "ticker" in df_up.columns:
        tickers = [str(x).strip().upper() for x in df_up["ticker"].dropna().tolist()]
        st.success(f"נטענו {len(tickers)} סימולים")
    else:
        st.error("בקובץ חייבת להיות עמודה בשם ticker")

tab1, tab2 = st.tabs(["סריקה", "ניתוח מניה"])

with tab1:
    if st.button("סרוק עכשיו", type="primary", use_container_width=True):
        if not tickers:
            st.warning("הכנס לפחות סימול אחד.")
        else:
            with st.spinner("מחשב איתותים..."):
                results = scan_universe(
                    tickers=tickers,
                    daily_period=daily_period,
                    use_hourly=use_hourly,
                    hourly_period=hourly_period,
                    horizon=horizon,
                    target_pct=target_pct,
                    min_turnover=min_turnover,
                )
            if results.empty:
                st.warning("לא התקבלו נתונים תקינים.")
            else:
                results["Signal"] = results["Score"].apply(
                    lambda s: "BUY CANDIDATE" if s >= min_score else ("WATCH" if s >= min_score-8 else "NO SIGNAL")
                )
                cols = [
                    "Ticker","Signal","Score","Price","DailyChangePct","VolumeRatio",
                    "VolumeZ","RSI14","ADX14","ATRpct","Breakout20Pct",
                    "HourlyScore","EmpiricalHitRate","BacktestN"
                ]
                show = results[[c for c in cols if c in results.columns]].copy()
                st.dataframe(
                    show.sort_values(["Signal","Score"], ascending=[True,False]),
                    use_container_width=True,
                    hide_index=True,
                )
                candidates = results[results["Score"] >= min_score].sort_values("Score", ascending=False)
                if len(candidates):
                    st.success(f"נמצאו {len(candidates)} מועמדים מעל ציון {min_score}.")
                else:
                    st.info("אין כרגע מניה שעברה את סף האיתות.")

with tab2:
    ticker = st.text_input("סימול", "1196.HK").strip().upper()
    if st.button("נתח מניה", use_container_width=True):
        with st.spinner("טוען ומנתח..."):
            daily = fetch_ohlcv(ticker, period=daily_period, interval="1d")
            if daily is None or len(daily) < 35:
                st.error("אין מספיק נתונים.")
            else:
                feat = compute_features(daily)
                latest = score_latest(feat, min_turnover=min_turnover)
                bt = backtest_signal(
                    feat,
                    horizon=horizon,
                    target_pct=target_pct,
                    score_threshold=max(55, min_score-5),
                    min_turnover=min_turnover,
                )
                hourly_score = None
                if use_hourly:
                    h = fetch_ohlcv(ticker, period=hourly_period, interval="1h")
                    if h is not None and len(h) >= 30:
                        hf = compute_features(h, intraday=True)
                        hourly_score = score_latest(hf, min_turnover=0)["score"]

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("ציון כמותי", f'{latest["score"]:.0f}/100')
                c2.metric("מחיר", f'{latest["price"]:.3f}')
                c3.metric("Volume Ratio", f'{latest["volume_ratio"]:.2f}x')
                c4.metric("RSI14", f'{latest["rsi14"]:.1f}')
                if hourly_score is not None:
                    st.metric("ציון שעתי", f"{hourly_score:.0f}/100")

                st.subheader("פירוק האיתות")
                explain = pd.DataFrame(latest["components"], columns=["Component","Points","Max"])
                st.dataframe(explain, use_container_width=True, hide_index=True)

                st.subheader("Walk‑Forward היסטורי")
                if bt["n"] > 0:
                    st.write(
                        f"ב-{bt['n']} מקרים היסטוריים שעמדו בסף, "
                        f"המחיר הגיע ליעד של {target_pct*100:.1f}% בתוך {horizon} ימים "
                        f"ב-{bt['hit_rate']*100:.1f}% מהמקרים."
                    )
                    st.caption(
                        "זהו שיעור הצלחה היסטורי בלבד. הוא אינו הסתברות מובטחת לעסקה הבאה."
                    )
                else:
                    st.info("אין מספיק אירועים היסטוריים דומים למדידת Hit Rate.")

                st.subheader("מחיר אחרון")
                chart = daily[["Close"]].tail(60)
                st.line_chart(chart)

st.markdown("---")
st.caption(
    "המודל נותן עדיפות לאיתות נדיר ואיכותי על פני הרבה איתותים. "
    "מומלץ לאמת כל שינוי במשקולות באמצעות Walk‑Forward ולא על אותה תקופה שעליה כווננו את המודל."
)
