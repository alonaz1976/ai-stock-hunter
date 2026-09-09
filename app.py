import streamlit as st
import pandas as pd
import yfinance as yf
from stock_hunter_v1 import analyze, backtest

st.set_page_config(page_title='AI Stock Hunter', page_icon='📈', layout='wide')
st.title('📈 AI Stock Hunter — V1')
st.caption('Smart Money • Entry Timing • Early Accumulation • Walk-forward Backtest')

with st.sidebar:
    st.header('Scanner settings')
    ticker = st.text_input('Ticker', 'NVDA').upper().strip() or 'TICKER'
    min_smart = st.slider('Minimum Smart Money', 0, 100, 70)
    min_entry = st.slider('Minimum Entry Score', 0, 100, 80)
    st.divider()
    st.caption('V1 uses OHLCV only. Valuation and fundamentals will be added in V2.')

st.subheader("Live market data")

period = st.selectbox(
    "History period",
    ["1y", "2y", "5y"],
    index=1
)

if st.button("Analyze stock", use_container_width=True):
    try:
        with st.spinner(f"Downloading market data for {ticker}..."):
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False
            )

        if df.empty:
            st.error("No market data found for this ticker.")
            st.stop()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        result = analyze(df, ticker)

    except Exception as e:
        st.error(f"Could not analyze {ticker}: {e}")
        st.stop()
else:
    st.info("Enter a ticker and press Analyze stock.")
    st.stop()

c1,c2,c3,c4 = st.columns(4)
c1.metric('Stock Score', f'{result.stock_score:.0f}/100')
c2.metric('Entry Score', f'{result.entry_score:.0f}/100')
c3.metric('Smart Money', f'{result.smart_money:.0f}/100')
c4.metric('Signal', result.verdict)

if result.early_accumulation:
    st.success('🐋 EARLY ACCUMULATION DETECTED')

st.subheader(f'{ticker} — Entry plan')
a,b,c,d = st.columns(4)
a.metric('Entry zone', f'${result.entry_low:,.2f} – ${result.entry_high:,.2f}')
b.metric('Invalidation', f'${result.invalidation:,.2f}')
c.metric('Target 1', f'${result.target1:,.2f}')
d.metric('Target 2', f'${result.target2:,.2f}')

st.subheader('Signal diagnostics')
diag = pd.DataFrame({
    'Metric':['RSI (14)','Relative Volume','CMF','Early Accumulation'],
    'Value':[result.rsi, f'{result.rvol:.2f}×', result.cmf, 'YES 🐋' if result.early_accumulation else 'No']
})
st.dataframe(diag, use_container_width=True, hide_index=True)

st.subheader('Price')
cols = {c.lower():c for c in df.columns}
if 'close' in cols:
    chart = df[[cols['close']]].copy()
    chart.columns=['Close']
    st.line_chart(chart)

st.subheader('Walk-forward backtest')
if st.button('Run backtest', type='primary'):
    with st.spinner('Testing historical signals without future leakage...'):
        signals, stats = backtest(df, ticker, min_smart=min_smart, min_entry=min_entry)
    if stats.get('signals',0) == 0:
        st.warning('No historical signals matched these thresholds.')
    else:
        st.write(f"Signals found: **{stats['signals']}**")
        metrics = []
        for h in (1,5,10,20):
            metrics.append({'Horizon':f'{h} trading days','Win rate':f"{stats.get(f'win_{h}d',0)*100:.1f}%",'Average return':f"{stats.get(f'avg_{h}d',0)*100:.2f}%"})
        st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)
        st.dataframe(signals.tail(50), use_container_width=True, hide_index=True)

st.caption('Research prototype — scores are quantitative signals, not guarantees or personalized investment advice.')
