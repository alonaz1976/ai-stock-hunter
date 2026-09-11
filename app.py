import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quant_engine import fetch_ohlcv, compute_features, score_latest, backtest_signal, scan_universe, entry_timing

APP_VERSION='V5.0 CLEAN'
st.set_page_config(page_title=f'AI Stock Hunter — {APP_VERSION}',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
st.markdown("""<style>
:root{--bg:#080b12;--panel:#111722;--panel2:#151d2b;--text:#f5f7fb;--muted:#8f9bad;--good:#35d49a;--warn:#f6c85f;--bad:#ff647c}
html,body,[data-testid='stAppViewContainer']{background:radial-gradient(circle at 15% 0%,#151a2e 0,#080b12 34%);color:var(--text)}
[data-testid='stHeader']{background:transparent}.block-container{max-width:1180px;padding-top:2rem;padding-bottom:5rem}
.hero{padding:26px 28px;border:1px solid #26334a;border-radius:24px;background:linear-gradient(135deg,rgba(124,92,255,.15),rgba(40,215,229,.05));margin-bottom:20px}
.hero-title{font-size:clamp(2.4rem,6vw,4.5rem);font-weight:850;line-height:.98}.hero-sub{color:var(--muted);font-size:1.08rem;margin-top:14px}.badge{display:inline-block;padding:5px 10px;border-radius:999px;background:#20283a;color:#b8c2d5;font-size:.78rem;font-weight:700}
.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid #263249;border-radius:20px;padding:18px 20px;margin-bottom:14px}.good{color:var(--good);font-weight:800}.warn{color:var(--warn);font-weight:800}.bad{color:var(--bad);font-weight:800}.muted{color:var(--muted)}.section{font-size:1.55rem;font-weight:800;margin:28px 0 12px}.stButton>button{width:100%;min-height:52px;border-radius:14px;background:linear-gradient(90deg,#6e56ff,#3c8cff);border:0;color:white;font-weight:750}
</style>""",unsafe_allow_html=True)
st.markdown(f"""<div class='hero'><span class='badge'>{APP_VERSION} • EARLY PREDICTION • ENTRY ENGINE</span><div class='hero-title'>📈 AI Stock Hunter<br>{APP_VERSION}</div><div class='hero-sub'>Smart Money • Early Accumulation • Precision Entry • Walk-forward Backtest • Confidence Engine</div></div>""",unsafe_allow_html=True)

DEFAULT_TICKERS='1196.HK,NVDA,NVMI,MU,AMD,AVGO,AMZN,META,GOOGL,MSFT,AAPL,TSLA,PLTR,CRWV,NBIS,GILD,ORCL,SMCI,ARM,TSM,QCOM,NFLX,UBER,COIN,HOOD,TTWO'

def signal(score,threshold,confidence=None):
    if score>=threshold and confidence!='LOW': return 'BUY CANDIDATE'
    if score>=threshold-8: return 'WATCH'
    return 'AVOID'

def cls(s): return 'good' if 'BUY' in s or 'ENTER' in s else 'warn' if 'WATCH' in s or 'WAIT' in s else 'bad'

def chart(df,title):
    q=df.tail(100); fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.76,.24],vertical_spacing=.04)
    fig.add_trace(go.Candlestick(x=q.index,open=q['Open'],high=q['High'],low=q['Low'],close=q['Close'],name='Price'),row=1,col=1)
    if 'ema20' in q.columns: fig.add_trace(go.Scatter(x=q.index,y=q['ema20'],name='EMA20'),row=1,col=1)
    if 'ema50' in q.columns: fig.add_trace(go.Scatter(x=q.index,y=q['ema50'],name='EMA50'),row=1,col=1)
    fig.add_trace(go.Bar(x=q.index,y=q['Volume'],name='Volume',opacity=.65),row=2,col=1)
    fig.update_layout(height=560,title=title,template='plotly_dark',xaxis_rangeslider_visible=False)
    return fig

tab1,tab2,tab3=st.tabs(['◉ Analyze','⌁ Opportunity Scanner','▦ Backtest Lab'])
with tab1:
    c1,c2,c3,c4=st.columns([1.3,1,1,1]); ticker=c1.text_input('Ticker','1196.HK'); hist=c2.selectbox('History',['3mo','6mo','1y'],1); horizon=c3.selectbox('Forecast days',[3,5,7,10],1); target=c4.selectbox('Target %',[3,5,6,8,10,15,20],2)
    buy_threshold=st.slider('BUY threshold',60,80,66)
    if st.button(f'Run {APP_VERSION} analysis'):
        t=ticker.strip().upper(); d=fetch_ohlcv(t,hist,'1d')
        if d is None or len(d)<35: st.error('Not enough market data.')
        else:
            f=compute_features(d); latest=score_latest(f,0)
            h=fetch_ohlcv(t,'1mo','1h'); hs=he=np.nan
            if h is not None and len(h)>=30:
                hl=score_latest(compute_features(h,True),0); hs=hl['score']; he=hl['early_score']
            qscore=latest['score'] if not np.isfinite(hs) else .78*latest['score']+.22*hs
            escore=latest['early_score'] if not np.isfinite(he) else .70*latest['early_score']+.30*he
            bt=backtest_signal(f,int(horizon),float(target)/100,buy_threshold,0)
            m15=fetch_ohlcv(t,'1mo','15m'); ef=compute_features(m15,True) if m15 is not None and len(m15)>=30 else (compute_features(h,True) if h is not None and len(h)>=30 else f)
            ent=entry_timing(ef,qscore,escore); sig=signal(qscore,buy_threshold,bt['confidence_label'])
            a,b,c,d1,e=st.columns(5); a.metric('Quant Score',f'{qscore:.1f}'); b.metric('Early Prediction',f'{escore:.1f}'); c.metric('Entry Score',f"{ent['entry_score']:.1f}"); d1.metric('Hit Rate',f"{bt['hit_rate']*100:.1f}%" if bt['n'] else '—'); e.metric('Signals',bt['n'])
            st.markdown(f"<div class='card'><div class='{cls(sig)}' style='font-size:1.3rem'>{sig}</div><div class='muted'>Confidence: <b>{bt['confidence_label']}</b> ({bt['confidence']:.0f}/100)</div><hr><div class='{cls(ent['status'])}'>{ent['status']}</div></div>",unsafe_allow_html=True)
            z1,z2,z3,z4=st.columns(4); z1.metric('Entry zone',f"{ent['zone_low']:.3f} – {ent['zone_high']:.3f}"); z2.metric('Breakout trigger',f"> {ent['trigger']:.3f}"); z3.metric('Invalidation',f"< {ent['invalidation']:.3f}"); z4.metric('Targets',f"{ent['target1']:.3f} / {ent['target2']:.3f}")
            st.plotly_chart(chart(f,f'{t} • Price / EMA / Volume'),use_container_width=True)

with tab2:
    tickers=st.text_area('Tickers',DEFAULT_TICKERS,height=150)
    c1,c2,c3,c4=st.columns(4); sh=c1.selectbox('History',['3mo','6mo','1y'],1,key='sh'); ho=c2.selectbox('Forecast days',[3,5,7,10],1,key='ho'); ta=c3.selectbox('Target %',[3,5,6,8,10],2,key='ta'); th=c4.number_input('BUY threshold',60,80,66)
    prefilter_top=st.slider('Deep-analysis finalists',10,50,30,5); only=st.checkbox('Show only BUY / WATCH',True)
    if st.button('Scan watchlist'):
        ts=[x.strip().upper() for x in tickers.split(',') if x.strip()]; res=scan_universe(ts,sh,True,'1mo',int(ho),float(ta)/100,0,int(th),int(prefilter_top))
        if res.empty: st.warning('No valid results returned.')
        else:
            res['Signal']=[signal(r.Score,int(th),r.Confidence) for _,r in res.iterrows()]
            if only: res=res[res.Signal.isin(['BUY CANDIDATE','WATCH'])]
            st.dataframe(res,use_container_width=True,hide_index=True)

with tab3:
    bt_t=st.text_input('Ticker for threshold test','QCOM'); bt_h=st.selectbox('History',['6mo','1y'],1,key='bth'); bt_days=st.selectbox('Forecast days',[3,5,7,10],1,key='btd'); bt_target=st.selectbox('Target %',[3,5,6,8,10],2,key='btt')
    if st.button('Compare thresholds'):
        d=fetch_ohlcv(bt_t.strip().upper(),bt_h,'1d')
        if d is None or len(d)<35: st.error('Not enough data')
        else:
            f=compute_features(d); rows=[]
            for x in [60,62,64,66,68,70,72,75]:
                b=backtest_signal(f,int(bt_days),float(bt_target)/100,x,0); rows.append({'Threshold':x,'Hit Rate %':round(b['hit_rate']*100,1) if b['n'] else np.nan,'Signals':b['n'],'Avg Return %':round(b['avg_return']*100,2) if b['n'] else np.nan,'Max Drawdown %':round(b['max_drawdown']*100,2) if b['n'] else np.nan,'Confidence':b['confidence_label'],'Confidence Score':b['confidence']})
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

st.caption('Research prototype. Quantitative estimates only.')
