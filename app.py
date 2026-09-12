import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quant_engine import fetch_ohlcv, compute_features, score_latest, backtest_signal, scan_universe, entry_timing, score_row, early_score_row

def early_event_backtest(feat, event_pct=.06, lookbacks=(1, 2, 3), min_turnover=0):
    """Single-ticker diagnostic: scores 1/2/3 trading days before close-to-close gains >= target."""
    f = feat.copy()
    if f is None or len(f) == 0:
        return pd.DataFrame()
    f["quant_score_bt"] = [score_row(r, min_turnover)[0] for _, r in f.iterrows()]
    f["early_score_bt"] = [early_score_row(r)[0] for _, r in f.iterrows()]
    f["day_return_bt"] = f["Close"].pct_change()
    rows = []
    for i in range(1, len(f)):
        day_ret = float(f["day_return_bt"].iloc[i]) if pd.notna(f["day_return_bt"].iloc[i]) else np.nan
        if not np.isfinite(day_ret) or day_ret < float(event_pct):
            continue
        row = {
            "EventDate": f.index[i],
            "EventReturnPct": day_ret * 100.0,
            "PrevClose": float(f["Close"].iloc[i-1]),
            "EventClose": float(f["Close"].iloc[i]),
        }
        for lb in lookbacks:
            j = i - int(lb)
            row[f"Quant_D{lb}"] = float(f["quant_score_bt"].iloc[j]) if j >= 0 else np.nan
            row[f"Early_D{lb}"] = float(f["early_score_bt"].iloc[j]) if j >= 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

st.set_page_config(page_title="AI Stock Hunter — V5.2.1",page_icon="📈",layout="wide",initial_sidebar_state="collapsed")
st.markdown("""<style>
:root{--bg:#080b12;--panel:#111722;--panel2:#151d2b;--text:#f5f7fb;--muted:#8f9bad;--accent:#7c5cff;--cyan:#28d7e5;--good:#35d49a;--warn:#f6c85f;--bad:#ff647c;--line:#243044}
html,body,[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 15% 0%,#151a2e 0,#080b12 34%);color:var(--text)} [data-testid="stHeader"]{background:transparent}.block-container{max-width:1180px;padding-top:2rem;padding-bottom:5rem} h1,h2,h3,h4,p,label,span,div{font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif}.hero{padding:26px 28px;border:1px solid #26334a;border-radius:24px;background:linear-gradient(135deg,rgba(124,92,255,.15),rgba(40,215,229,.05));box-shadow:0 18px 50px rgba(0,0,0,.22);margin-bottom:20px}.hero-title{font-size:clamp(2.4rem,6vw,4.5rem);font-weight:850;line-height:.98;letter-spacing:-.05em}.hero-sub{color:var(--muted);font-size:1.08rem;margin-top:14px}.badge{display:inline-block;padding:5px 10px;border-radius:999px;background:#20283a;color:#b8c2d5;font-size:.78rem;font-weight:700;letter-spacing:.04em}.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid #263249;border-radius:20px;padding:18px 20px;margin-bottom:14px;box-shadow:0 10px 28px rgba(0,0,0,.16)}.good{color:var(--good);font-weight:800}.warn{color:var(--warn);font-weight:800}.bad{color:var(--bad);font-weight:800}.muted{color:var(--muted)}.section{font-size:1.55rem;font-weight:800;margin:28px 0 12px}.score{font-size:2.1rem;font-weight:850}.stButton>button{width:100%;min-height:52px;border-radius:14px;background:linear-gradient(90deg,#6e56ff,#3c8cff);border:0;color:white;font-weight:750}.stButton>button:hover{filter:brightness(1.08);color:white}.stTextArea textarea,.stTextInput input,div[data-baseweb="select"]>div{background:#111722!important;border-color:#29354a!important;border-radius:13px!important}.stDataFrame{border:1px solid #263249;border-radius:16px;overflow:hidden}[data-testid="stMetric"]{background:#111722;border:1px solid #263249;padding:14px;border-radius:16px}[data-testid="stMetricValue"]{font-size:1.55rem}.stTabs [data-baseweb="tab-list"]{gap:10px}.stTabs [data-baseweb="tab"]{border-radius:12px;padding:8px 14px}.stTabs [aria-selected="true"]{background:#171f31}
@media (max-width: 700px){
  .block-container{padding-left:.85rem;padding-right:.85rem;padding-top:1rem}
  .stTabs [data-baseweb="tab-list"]{gap:2px;width:100%;overflow:visible}
  .stTabs [data-baseweb="tab"]{flex:1 1 0;min-width:0;padding:7px 4px;font-size:.82rem;white-space:nowrap;justify-content:center}
  .stTabs [data-baseweb="tab"] p{font-size:.82rem!important;white-space:nowrap!important}
}

</style>""",unsafe_allow_html=True)
st.markdown("""<div class='hero'><span class='badge'>V5.2.1 • IMPORT HOTFIX • EARLY EVENT BACKTEST</span><div class='hero-title'>📈 AI Stock Hunter<br>V5.2.1</div><div class='hero-sub'>Smart Money • Early Accumulation • Precision Entry • Walk-forward Backtest • Confidence Engine</div></div>""",unsafe_allow_html=True)

DEFAULT_TICKERS="1196.HK,NVDA,NVMI,MU,AMD,AVGO,AMZN,META,GOOGL,MSFT,AAPL,TSLA,PLTR,CRWV,NBIS,GILD,ORCL,SMCI,ARM,TSM,QCOM,NFLX,UBER,COIN,HOOD,TTWO"
BROAD_200="1196.HK,NVDA,NVMI,MU,AMD,AVGO,AMZN,META,GOOGL,MSFT,AAPL,TSLA,PLTR,CRWV,NBIS,GILD,ORCL,SMCI,ARM,TSM,QCOM,NFLX,UBER,COIN,HOOD,TTWO,ADBE,CRM,INTC,CSCO,AMAT,LRCX,KLAC,MRVL,ADI,TXN,MCHP,ON,MPWR,DELL,HPE,ANET,NET,DDOG,SNOW,ZS,CRWD,PANW,FTNT,OKTA,SHOP,MELI,SE,ABNB,BKNG,DASH,RBLX,SPOT,SNAP,PINS,ROKU,PYPL,XYZ,V,MA,JPM,BAC,WFC,C,GS,MS,AXP,BLK,SCHW,COF,USB,PNC,TFC,BK,STT,BRK-B,UNH,LLY,JNJ,ABBV,MRK,PFE,AMGN,REGN,VRTX,ISRG,MDT,SYK,BSX,EW,TMO,DHR,ABT,MDLZ,KO,PEP,PG,COST,WMT,TGT,HD,LOW,NKE,SBUX,MCD,CMG,YUM,DIS,CMCSA,T,VZ,TMUS,CHTR,XOM,CVX,COP,SLB,EOG,OXY,MPC,VLO,PSX,KMI,WMB,NEE,DUK,SO,AEP,SRE,EXC,D,CEG,VST,CAT,DE,GE,GEV,HON,RTX,LMT,NOC,BA,GD,ETN,EMR,PH,MMM,UPS,FDX,UNP,CSX,NSC,WM,RSG,LIN,APD,SHW,FCX,NEM,NUE,STLD,AA,DOW,DD,GM,F,TM,RIVN,LCID,APO,KKR,BX,ARES,SPGI,MCO,CME,ICE,NDAQ,CB,MMC,AON,PGR,ALL,MET,PRU,AFL,CI,CVS,HUM,CNC,ELV,HCA,IQV,ZTS,BIIB"
def safe(x,d=2):
    try:return "—" if pd.isna(x) else f"{float(x):.{d}f}"
    except:return "—"
def signal(score,threshold,confidence=None):
    if score>=threshold and confidence!="LOW":return "BUY CANDIDATE"
    if score>=threshold-8:return "WATCH"
    return "AVOID"
def cls(s):return "good" if "BUY" in s or "ENTER" in s else "warn" if "WATCH" in s or "WAIT" in s else "bad"
def chart(df,title):
    q=df.tail(100); fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.76,.24],vertical_spacing=.04)
    fig.add_trace(go.Candlestick(x=q.index,open=q.Open,high=q.High,low=q.Low,close=q.Close,name="Price"),row=1,col=1)
    if "ema20" in q: fig.add_trace(go.Scatter(x=q.index,y=q.ema20,name="EMA20",line=dict(width=1.4)),row=1,col=1)
    if "ema50" in q: fig.add_trace(go.Scatter(x=q.index,y=q.ema50,name="EMA50",line=dict(width=1.2)),row=1,col=1)
    fig.add_trace(go.Bar(x=q.index,y=q.Volume,name="Volume",opacity=.65),row=2,col=1)
    fig.update_layout(height=570,title=title,template="plotly_dark",paper_bgcolor="#0b0f17",plot_bgcolor="#0b0f17",margin=dict(l=8,r=8,t=45,b=8),xaxis_rangeslider_visible=False,legend_orientation="h")
    return fig

tab1,tab2,tab3=st.tabs(["◉ Analyze","⌁ Scanner","▦ Backtest"])
with tab1:
    c1,c2,c3,c4=st.columns([1.3,1,1,1]); ticker=c1.text_input("Ticker","1196.HK"); hist=c2.selectbox("History",["3mo","6mo","1y"],1); horizon=c3.selectbox("Forecast days",[3,5,7,10],1); target=c4.selectbox("Target %",[3,5,6,8,10,15,20],2)
    buy_threshold=st.slider("BUY threshold",60,80,66)
    if st.button("Run V5.2.1 analysis"):
        t=ticker.strip().upper()
        with st.spinner("Running Quant + Early Prediction + Entry Engine..."):
            d=fetch_ohlcv(t,hist,"1d")
            if d is None or len(d)<35: st.error("Not enough market data.")
            else:
                f=compute_features(d); latest=score_latest(f,0)
                h=fetch_ohlcv(t,"1mo","1h"); hs=he=np.nan
                if h is not None and len(h)>=30:
                    hl=score_latest(compute_features(h,True),0); hs=hl["score"]; he=hl["early_score"]
                qscore=latest["score"] if not np.isfinite(hs) else .78*latest["score"]+.22*hs
                escore=latest["early_score"] if not np.isfinite(he) else .70*latest["early_score"]+.30*he
                bt=backtest_signal(f,int(horizon),float(target)/100,buy_threshold,0)
                # 15m precision layer where Yahoo supports it
                m15=fetch_ohlcv(t,"1mo","15m"); ef=compute_features(m15,True) if m15 is not None and len(m15)>=30 else (compute_features(h,True) if h is not None and len(h)>=30 else f)
                ent=entry_timing(ef,qscore,escore); sig=signal(qscore,buy_threshold,bt["confidence_label"])
                st.markdown("<div class='section'>Decision cockpit</div>",unsafe_allow_html=True)
                a,b,c,d1,e=st.columns(5); a.metric("Quant Score",f"{qscore:.1f}"); b.metric("Early Prediction",f"{escore:.1f}"); c.metric("Entry Score",f"{ent['entry_score']:.1f}"); d1.metric("Hit Rate",f"{bt['hit_rate']*100:.1f}%" if bt['n'] else "—"); e.metric("Signals",bt['n'])
                st.markdown(f"<div class='card'><div style='display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap'><div><div class='{cls(sig)}' style='font-size:1.3rem'>{sig}</div><div class='muted'>Confidence: <b>{bt['confidence_label']}</b> ({bt['confidence']:.0f}/100)</div></div><div><div class='{cls(ent['status'])}' style='font-size:1.2rem'>{ent['status']}</div><div class='muted'>Precision Entry Engine • 15m/1h confirmation</div></div></div></div>",unsafe_allow_html=True)
                st.markdown("<div class='section'>Precision entry</div>",unsafe_allow_html=True)
                z1,z2,z3,z4=st.columns(4); z1.metric("Entry zone",f"{ent['zone_low']:.3f} – {ent['zone_high']:.3f}"); z2.metric("Breakout trigger",f"> {ent['trigger']:.3f}"); z3.metric("Invalidation",f"< {ent['invalidation']:.3f}"); z4.metric("Targets",f"{ent['target1']:.3f} / {ent['target2']:.3f}")
                st.plotly_chart(chart(f,f"{t} • Price / EMA / Volume"),use_container_width=True)
                st.markdown("<div class='section'>Why this stock?</div>",unsafe_allow_html=True)
                reasons=[]
                if latest['obv_slope5']>0:reasons.append("OBV is rising — accumulation pressure")
                if latest['cmf20']>0:reasons.append(f"CMF positive ({latest['cmf20']:.2f}) — buying flow")
                if latest['squeeze']:reasons.append("Bollinger/Keltner squeeze is active")
                if latest['squeeze_release']:reasons.append("Squeeze release detected")
                if latest['pv_divergence']:reasons.append("Positive price/volume divergence")
                if latest['roc_accel']>0:reasons.append("ROC is accelerating")
                st.markdown("<div class='card'>"+("<br>".join("✓ "+x for x in reasons) if reasons else "No strong early-accumulation confirmation yet.")+"</div>",unsafe_allow_html=True)
                left,right=st.columns(2)
                with left:
                    st.markdown("#### Quant components"); comp=pd.DataFrame(latest['components'],columns=['Component','Points','Max']); comp['Strength %']=(100*comp.Points/comp.Max).round(); st.dataframe(comp,use_container_width=True,hide_index=True)
                with right:
                    st.markdown("#### Early Prediction components"); ec=pd.DataFrame(latest['early_components'],columns=['Component','Points','Max']); ec['Strength %']=(100*ec.Points/ec.Max).round(); st.dataframe(ec,use_container_width=True,hide_index=True)
                st.markdown("<div class='section'>Walk-forward evidence</div>",unsafe_allow_html=True)
                b1,b2,b3,b4=st.columns(4); b1.metric("Hit Rate",f"{bt['hit_rate']*100:.1f}%" if bt['n'] else "—"); b2.metric("Sample Size",bt['n']); b3.metric("Avg fwd return",f"{bt['avg_return']*100:.2f}%" if bt['n'] else "—"); b4.metric("Max drawdown",f"{bt['max_drawdown']*100:.2f}%" if bt['n'] else "—")
                st.caption(f"Hit = price reached +{target}% within {horizon} trading days after a historical score ≥ {buy_threshold}. Sample size and confidence are shown separately so a high hit rate on very few events is not treated as strong evidence.")
                st.markdown("<div class='section'>Early Prediction event study</div>",unsafe_allow_html=True)
                st.caption(f"Single-stock diagnostic: finds every trading day in the selected {hist} history where the CLOSE rose at least +{target}% versus the previous CLOSE, then shows Quant and Early Prediction 1, 2 and 3 trading days BEFORE that move.")
                ev=early_event_backtest(f,float(target)/100,(1,2,3),0)
                if ev.empty:
                    st.info(f"No single-day close-to-close moves of +{target}% or more were found in this history window.")
                else:
                    show=ev.copy()
                    show['EventDate']=pd.to_datetime(show['EventDate']).dt.strftime('%Y-%m-%d')
                    show=show.rename(columns={
                        'EventDate':'Event date','EventReturnPct':'Move %',
                        'Early_D3':'Early -3d','Quant_D3':'Quant -3d',
                        'Early_D2':'Early -2d','Quant_D2':'Quant -2d',
                        'Early_D1':'Early -1d','Quant_D1':'Quant -1d',
                        'PrevClose':'Previous close','EventClose':'Event close'
                    })
                    order=['Event date','Move %','Early -3d','Quant -3d','Early -2d','Quant -2d','Early -1d','Quant -1d','Previous close','Event close']
                    for col in ['Move %','Early -3d','Quant -3d','Early -2d','Quant -2d','Early -1d','Quant -1d','Previous close','Event close']:
                        if col in show: show[col]=pd.to_numeric(show[col],errors='coerce').round(1 if 'close' not in col.lower() else 3)
                    st.metric(f"+{target}% single-day events",len(show))
                    st.dataframe(show[[c for c in order if c in show]],use_container_width=True,hide_index=True)
                    early_cols=[c for c in ['Early -3d','Early -2d','Early -1d'] if c in show]
                    quant_cols=[c for c in ['Quant -3d','Quant -2d','Quant -1d'] if c in show]
                    eavg=float(show[early_cols].stack().mean()) if early_cols else np.nan
                    qavg=float(show[quant_cols].stack().mean()) if quant_cols else np.nan
                    x1,x2,x3=st.columns(3)
                    x1.metric("Avg Early before move",f"{eavg:.1f}" if np.isfinite(eavg) else "—")
                    x2.metric("Avg Quant before move",f"{qavg:.1f}" if np.isfinite(qavg) else "—")
                    x3.metric("Events found",len(show))
                    st.caption("This table is diagnostic, not a buy signal. It helps us see whether Early Prediction tends to rise BEFORE strong days and whether its formula/weights need adjustment.")
with tab2:
    st.markdown("<div class='section'>Multi-stock opportunity scanner</div>",unsafe_allow_html=True)
    universe_mode=st.radio("Universe",["Broad 200 (recommended)","Core 26","Custom"],horizontal=True)
    default_universe=BROAD_200 if universe_mode.startswith("Broad") else DEFAULT_TICKERS
    tickers=st.text_area("Tickers",default_universe,disabled=universe_mode!="Custom",height=150)
    c1,c2,c3,c4=st.columns(4); sh=c1.selectbox("History",["3mo","6mo","1y"],1,key="sh"); ho=c2.selectbox("Forecast days",[3,5,7,10],1,key="ho"); ta=c3.selectbox("Target %",[3,5,6,8,10],2,key="ta"); th=c4.number_input("BUY threshold",60,80,66)
    prefilter_top=st.slider("Deep-analysis finalists",10,50,30,5,help="All tickers get the fast daily pass. Only the strongest finalists get 1h + 15m + entry timing + full backtest.")
    st.caption(f"Stage 1 scans {len([x for x in tickers.split(chr(44)) if x.strip()])} tickers on daily data. Stage 2 performs deep intraday analysis only on the top {prefilter_top}.")
    only=st.checkbox("Show only BUY / WATCH",True)
    if st.button("Scan watchlist"):
        ts=[x.strip().upper() for x in tickers.split(',') if x.strip()]
        with st.spinner("Scanning Quant V4 universe..."): res=scan_universe(ts,sh,True,"1mo",int(ho),float(ta)/100,0,int(th),int(prefilter_top))
        if res.empty:st.warning("No valid results returned.")
        else:
            res['Signal']=[signal(r.Score,int(th),r.Confidence) for _,r in res.iterrows()]
            if only:res=res[res.Signal.isin(['BUY CANDIDATE','WATCH'])]
            st.session_state.v4res=res; st.session_state.v4time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if 'v4res' in st.session_state:
        res=st.session_state.v4res
        st.caption("Last scan: "+st.session_state.v4time)
        for _,r in res.head(5).iterrows():
            st.markdown(f"<div class='card'><div style='display:flex;justify-content:space-between'><div><b style='font-size:1.3rem'>{r.Ticker}</b><div class='{cls(r.Signal)}'>{r.Signal}</div></div><div style='text-align:right'><span class='score'>{r.Score:.1f}</span><div class='muted'>Quant Score</div></div></div><hr style='border-color:#263249'><div class='muted'>Early {r.EarlyScore:.1f} • RVOL {safe(r.VolumeRatio)}x • RSI {safe(r.RSI14,1)} • Hit {safe(r.EmpiricalHitRate,1)}% ({int(r.BacktestN)} signals) • Confidence {r.Confidence}</div></div>",unsafe_allow_html=True)
        cols=['Ticker','Signal','Score','EarlyScore','EntryScore','EntryStatus','EntryLow','EntryHigh','Trigger','Invalidation','Price','VolumeRatio','RSI14','ADX14','CMF20','EmpiricalHitRate','BacktestN','Confidence','ConfidenceScore','AvgReturn','MaxDrawdown']
        st.dataframe(res[[c for c in cols if c in res]],use_container_width=True,hide_index=True)
with tab3:
    st.markdown("<div class='section'>Threshold Backtest Lab</div>",unsafe_allow_html=True)
    bt_t=st.text_input("Ticker for threshold test","QCOM"); bt_h=st.selectbox("History",["6mo","1y"],1,key="bth"); bt_days=st.selectbox("Forecast days",[3,5,7,10],1,key="btd"); bt_target=st.selectbox("Target %",[3,5,6,8,10],2,key="btt")
    if st.button("Compare thresholds"):
        d=fetch_ohlcv(bt_t.strip().upper(),bt_h,"1d")
        if d is None or len(d)<35:st.error("Not enough data")
        else:
            f=compute_features(d); rows=[]
            for x in [60,62,64,66,68,70,72,75]:
                b=backtest_signal(f,int(bt_days),float(bt_target)/100,x,0); rows.append({'Threshold':x,'Hit Rate %':round(b['hit_rate']*100,1) if b['n'] else np.nan,'Signals':b['n'],'Avg Return %':round(b['avg_return']*100,2) if b['n'] else np.nan,'Max Drawdown %':round(b['max_drawdown']*100,2) if b['n'] else np.nan,'Confidence':b['confidence_label'],'Confidence Score':b['confidence']})
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
            st.caption("Use this table to choose a threshold from evidence, not from a fixed arbitrary number. Prefer a balance of hit rate, sample size, return and drawdown.")
st.caption("Research prototype. Scores and entry zones are quantitative estimates, not guarantees or personalized investment advice.")
