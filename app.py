import streamlit as st
import pandas as pd
import yfinance as yf
from stock_hunter_v1 import analyze, backtest

st.set_page_config(page_title='AI Stock Hunter', page_icon='📈', layout='wide')
st.title('📈 AI Stock Hunter — V2')
st.caption('Smart Money • Entry Timing • Early Accumulation • Multi-stock Scanner')

DEFAULT_TICKERS = [
    'NVDA','NVMI','MU','AMD','AVGO','AMZN','META','GOOGL','MSFT','AAPL',
    'TSLA','PLTR','CRWV','NBIS','GILD','ORCL','SMCI','ARM','TSM','QCOM',
    'NFLX','UBER','COIN','HOOD','TTWO'
]

@st.cache_data(ttl=900, show_spinner=False)
def download_one(ticker, period='2y'):
    df = yf.download(ticker, period=period, interval='1d', auto_adjust=False, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.reset_index()

@st.cache_data(ttl=900, show_spinner=False)
def scan_tickers(tickers, period='2y'):
    rows=[]
    errors=[]
    for t in tickers:
        try:
            df=download_one(t,period)
            if df.empty:
                errors.append(t); continue
            r=analyze(df,t)
            rows.append({
                'Ticker':t,
                'Price':r.price,
                'Stock':r.stock_score,
                'Entry':r.entry_score,
                'Smart Money':r.smart_money,
                'RSI':r.rsi,
                'RVOL':r.rvol,
                'CMF':r.cmf,
                'Early Accumulation':'🐋 YES' if r.early_accumulation else 'No',
                'Signal':r.verdict,
                'Entry Low':r.entry_low,
                'Entry High':r.entry_high,
                'Invalidation':r.invalidation,
                'Target 1':r.target1,
                'Target 2':r.target2,
            })
        except Exception:
            errors.append(t)
    out=pd.DataFrame(rows)
    if not out.empty:
        out['Opportunity']=(0.50*out['Entry']+0.35*out['Smart Money']+0.15*out['Stock']).round(1)
        out=out.sort_values(['Opportunity','Entry','Smart Money'],ascending=False).reset_index(drop=True)
    return out, errors

with st.sidebar:
    st.header('Scanner settings')
    ticker = st.text_input('Ticker', 'NVDA').upper().strip() or 'NVDA'
    min_smart = st.slider('Minimum Smart Money', 0, 100, 70)
    min_entry = st.slider('Minimum Entry Score', 0, 100, 80)
    st.divider()
    st.caption('V2 scanner uses daily OHLCV. Valuation/fundamentals are the next layer.')

single_tab, scan_tab = st.tabs(['🔎 Analyze one stock','🔥 Scan watchlist'])

with single_tab:
    st.subheader('Market data')
    period=st.selectbox('History period',['1y','2y','5y'],index=1,key='single_period')
    if st.button('Analyze stock',use_container_width=True,key='analyze_one'):
        try:
            with st.spinner(f'Downloading {ticker} market data...'):
                df=download_one(ticker,period)
            if df.empty:
                st.error('No market data found.'); st.stop()
            result=analyze(df,ticker)
            st.session_state['single_result']=(result,df)
        except Exception as e:
            st.error(f'Could not analyze {ticker}: {e}')

    if 'single_result' in st.session_state:
        result,df=st.session_state['single_result']
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Stock Score',f'{result.stock_score:.0f}/100')
        c2.metric('Entry Score',f'{result.entry_score:.0f}/100')
        c3.metric('Smart Money',f'{result.smart_money:.0f}/100')
        c4.metric('Signal',result.verdict)
        if result.early_accumulation:
            st.success('🐋 EARLY ACCUMULATION DETECTED')
        st.subheader(f'{result.ticker} — Entry plan')
        a,b,c,d=st.columns(4)
        a.metric('Entry zone',f'${result.entry_low:,.2f} – ${result.entry_high:,.2f}')
        b.metric('Invalidation',f'${result.invalidation:,.2f}')
        c.metric('Target 1',f'${result.target1:,.2f}')
        d.metric('Target 2',f'${result.target2:,.2f}')
        diag=pd.DataFrame({
            'Metric':['Price','RSI (14)','Relative Volume','CMF','EMA20','EMA50','EMA200','MACD','Early Accumulation'],
            'Value':[f'${result.price:,.2f}',result.rsi,f'{result.rvol:.2f}×',result.cmf,result.ema20,result.ema50,result.ema200,'Bullish' if result.macd_bullish else 'Bearish','YES 🐋' if result.early_accumulation else 'No']
        })
        st.dataframe(diag,use_container_width=True,hide_index=True)
        cols={str(c).lower():c for c in df.columns}
        if 'close' in cols:
            chart=df[[cols['close']]].copy(); chart.columns=['Close']; st.line_chart(chart)
        with st.expander('Walk-forward backtest'):
            if st.button('Run backtest',type='primary',key='bt'):
                with st.spinner('Testing historical signals...'):
                    signals,stats=backtest(df,result.ticker,min_smart=min_smart,min_entry=min_entry)
                if stats.get('signals',0)==0:
                    st.warning('No historical signals matched these thresholds.')
                else:
                    st.write(f"Signals found: **{stats['signals']}**")
                    metrics=[]
                    for h in (1,5,10,20):
                        metrics.append({'Horizon':f'{h} days','Win rate':f"{stats.get(f'win_{h}d',0)*100:.1f}%",'Average return':f"{stats.get(f'avg_{h}d',0)*100:.2f}%"})
                    st.dataframe(pd.DataFrame(metrics),use_container_width=True,hide_index=True)

with scan_tab:
    st.subheader('Multi-stock opportunity scanner')
    st.caption('Scans a focused liquid watchlist first. We can expand to the full NASDAQ/NYSE after validating speed and signal quality.')
    custom=st.text_area('Tickers to scan (comma separated)',','.join(DEFAULT_TICKERS),height=110)
    scan_period=st.selectbox('History period',['1y','2y'],index=1,key='scan_period')
    only_candidates=st.checkbox('Show only candidates',value=False,help='Entry ≥ minimum Entry Score OR Smart Money ≥ minimum Smart Money')
    if st.button('🚀 Scan now',type='primary',use_container_width=True,key='scan_now'):
        tickers=list(dict.fromkeys([x.strip().upper() for x in custom.replace('\n',',').split(',') if x.strip()]))[:60]
        progress=st.progress(0,text='Starting scan...')
        # progress bar is visual only because cached scanner runs as one operation
        progress.progress(20,text=f'Scanning {len(tickers)} stocks...')
        with st.spinner('Downloading data and ranking opportunities...'):
            table,errors=scan_tickers(tickers,scan_period)
        progress.progress(100,text='Scan complete')
        st.session_state['scan_result']=(table,errors)

    if 'scan_result' in st.session_state:
        table,errors=st.session_state['scan_result']
        if table.empty:
            st.error('No stocks could be analyzed.')
        else:
            shown=table.copy()
            if only_candidates:
                shown=shown[(shown['Entry']>=min_entry)|(shown['Smart Money']>=min_smart)]
            top=table.iloc[0]
            st.success(f"Top opportunity: {top['Ticker']} | Opportunity {top['Opportunity']:.0f}/100 | Entry {top['Entry']:.0f} | Smart Money {top['Smart Money']:.0f}")
            st.markdown('### 🔥 Ranked opportunities')
            main_cols=['Ticker','Price','Opportunity','Entry','Smart Money','Stock','RSI','RVOL','CMF','Early Accumulation','Signal']
            st.dataframe(shown[main_cols],use_container_width=True,hide_index=True)
            st.markdown('### 🎯 Entry plans — Top 10')
            plans=shown.head(10)[['Ticker','Entry Low','Entry High','Invalidation','Target 1','Target 2','Signal']]
            st.dataframe(plans,use_container_width=True,hide_index=True)
            if errors:
                st.caption('Could not load: '+', '.join(errors))

st.caption('Research prototype — quantitative signals are not guarantees or personalized investment advice.')
