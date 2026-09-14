import streamlit as st
import pandas as pd
import numpy as np
import math
import importlib.util
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

APP_VERSION = "5.7.1"
APP_BUILD_ID = "V571-SCANNER50-20260914-A"

# Always load the quant engine from the quant_engine.py file that sits next to
# this app.py.  Using a unique module name deliberately bypasses a stale
# sys.modules entry that Streamlit can keep alive across hot-reloads.
def _load_local_quant_engine():
    engine_path = Path(__file__).resolve().with_name("quant_engine.py")
    if not engine_path.exists():
        raise FileNotFoundError(f"Missing quant engine file: {engine_path}")
    unique_name = f"ai_stock_hunter_quant_{APP_BUILD_ID.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(unique_name, engine_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {engine_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

qe = _load_local_quant_engine()
# Bind engine functions defensively so an out-of-sync quant_engine.py does not crash
# the whole Streamlit app during import. The UI below will show an explicit sync error
# if any required V5.4.x function is missing.
fetch_ohlcv = getattr(qe, "fetch_ohlcv", None)
compute_features = getattr(qe, "compute_features", None)
score_latest = getattr(qe, "score_latest", None)
backtest_signal = getattr(qe, "backtest_signal", None)
scan_universe_dynamic = getattr(qe, "scan_universe_dynamic", None)
entry_timing = getattr(qe, "entry_timing", None)
score_row = getattr(qe, "score_row", None)
early_score_row = getattr(qe, "early_score_row", None)
dynamic_scores = getattr(qe, "dynamic_scores", None)
compare_static_dynamic_backtest = getattr(qe, "compare_static_dynamic_backtest", None)


# -----------------------------------------------------------------------------
# V5.5 robust dynamic-calibration fallback
# The dynamic layer is duplicated locally on purpose. If Streamlit ever serves an
# older cached quant_engine.py during a deploy, Analyze/Scanner/Backtest still work
# instead of stopping with an incompatibility error.
# -----------------------------------------------------------------------------
def _v544_clamp(x, lo=0.0, hi=100.0):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def _v544_component_dict(r, kind='early', min_turnover=0):
    comps = early_score_row(r)[1] if kind == 'early' else score_row(r, min_turnover)[1]
    return {name: (float(pts), float(mx)) for name, pts, mx in comps}


def _v544_calibrate_components(feat, kind='early', event_pct=.03, lookbacks=(1,2,3), min_turnover=0):
    f = feat.copy()
    if f is None or len(f) < 35:
        return {'events':0,'confidence':0.0,'multipliers':{},'table':pd.DataFrame()}
    event_ret = f['Close'].pct_change()
    events = [i for i in range(1,len(f)) if pd.notna(event_ret.iloc[i]) and float(event_ret.iloc[i]) >= float(event_pct)]
    sample_rel = min(1.0, math.sqrt(len(events)/30.0)) if events else 0.0
    baseline, maxima = {}, {}
    for _, r in f.iterrows():
        for name,(pts,mx) in _v544_component_dict(r,kind,min_turnover).items():
            maxima[name]=mx
            baseline.setdefault(name,[]).append(pts>0)
    rows=[]; multipliers={}
    for name,mx in maxima.items():
        pre=[]; seen=set(); by_lb={lb:[] for lb in lookbacks}
        for eno,i in enumerate(events):
            for lb in lookbacks:
                j=i-int(lb)
                if j<0: continue
                active=_v544_component_dict(f.iloc[j],kind,min_turnover).get(name,(0,mx))[0] > 0
                pre.append(active); by_lb[lb].append(active)
                if active: seen.add(eno)
        pre_rate=float(np.mean(pre)) if pre else np.nan
        base_rate=float(np.mean(baseline.get(name,[]))) if baseline.get(name) else np.nan
        lift=pre_rate/base_rate if np.isfinite(pre_rate) and np.isfinite(base_rate) and base_rate>0 else 1.0
        coverage=len(seen)/len(events) if events else 0.0
        lb_rates=[float(np.mean(v)) for v in by_lb.values() if v]
        stability=max(0.55,1.0-(float(np.std(lb_rates))/0.35)) if lb_rates else 0.55
        evidence=sample_rel*(0.55+0.45*coverage)*stability
        mult=float(np.clip(1.0 + evidence*(lift-1.0)*1.65,0.45,1.80))
        multipliers[name]=mult
        rows.append({'Component':name,'Base Weight':mx,'Dynamic Weight':mx*mult,'Lift x':lift,'Coverage %':coverage*100,'Sample Reliability %':sample_rel*100,'Stability %':stability*100,'Multiplier':mult})
    table=pd.DataFrame(rows)
    if not table.empty:
        table=table.sort_values(['Dynamic Weight','Lift x'],ascending=False).reset_index(drop=True)
    return {'events':len(events),'confidence':sample_rel*100,'multipliers':multipliers,'table':table}


def _v544_dynamic_score_row(r, calibration, kind='early', min_turnover=0):
    comps=_v544_component_dict(r,kind,min_turnover)
    mults=(calibration or {}).get('multipliers',{})
    num=den=0.0; details=[]
    for name,(pts,mx) in comps.items():
        m=float(mults.get(name,1.0)); dw=mx*m
        strength=(pts/mx) if mx>0 else 0.0
        num += strength*dw; den += dw
        details.append((name,pts,mx,dw,m))
    score=100.0*num/den if den>0 else 0.0
    return float(_v544_clamp(score)), details


def _v544_calibrate_combinations(feat,event_pct=.03,lookbacks=(1,2,3),min_turnover=0,top_n=12):
    f=feat.copy()
    if f is None or len(f)<35: return pd.DataFrame()
    er=f['Close'].pct_change()
    events=[i for i in range(1,len(f)) if pd.notna(er.iloc[i]) and float(er.iloc[i])>=float(event_pct)]
    if not events: return pd.DataFrame()
    def active_set(r):
        e={f'E:{n}' for n,(p,m) in _v544_component_dict(r,'early',min_turnover).items() if p>0}
        q={f'Q:{n}' for n,(p,m) in _v544_component_dict(r,'quant',min_turnover).items() if p>0}
        return e|q
    sets=[active_set(r) for _,r in f.iterrows()]
    names=sorted(set().union(*sets)) if sets else []
    pairs=[]
    for a_i,a in enumerate(names):
        for b in names[a_i+1:]:
            if a[0]==b[0]: continue
            base=np.mean([(a in ss and b in ss) for ss in sets])
            if base<=0: continue
            pre=[]; seen=set()
            for eno,i in enumerate(events):
                for lb in lookbacks:
                    j=i-int(lb)
                    if j<0: continue
                    on=a in sets[j] and b in sets[j]
                    pre.append(on)
                    if on: seen.add(eno)
            pr=np.mean(pre) if pre else 0.0
            lift=pr/base if base else np.nan
            cov=100*len(seen)/len(events)
            if np.isfinite(lift):
                pairs.append({'Combination':a[2:]+' + '+b[2:],'Lift x':lift,'Coverage %':cov,'Pre-event active %':100*pr,'Baseline active %':100*base})
    z=pd.DataFrame(pairs)
    if z.empty: return z
    return z.sort_values(['Lift x','Coverage %'],ascending=False).head(top_n).reset_index(drop=True)


def _v544_dynamic_scores(feat,event_pct=.03,min_turnover=0):
    r=feat.dropna(subset=['Close']).iloc[-1]
    ec=_v544_calibrate_components(feat,'early',event_pct,(1,2,3),min_turnover)
    qc=_v544_calibrate_components(feat,'quant',event_pct,(1,2,3),min_turnover)
    se=early_score_row(r)[0]; sq=score_row(r,min_turnover)[0]
    de,ed=_v544_dynamic_score_row(r,ec,'early',min_turnover)
    dq,qd=_v544_dynamic_score_row(r,qc,'quant',min_turnover)
    er=float(ec.get('confidence',0))/100.0; qr=float(qc.get('confidence',0))/100.0
    early_w=float(np.clip(.35+.10*(er-qr),.25,.45)); quant_w=1.0-early_w
    pred=quant_w*dq+early_w*de
    return {'static_early':se,'static_quant':sq,'dynamic_early':de,'dynamic_quant':dq,'final_prediction':float(_v544_clamp(pred)),'early_weight':early_w,'quant_weight':quant_w,'early_calibration':ec,'quant_calibration':qc,'combinations':_v544_calibrate_combinations(feat,event_pct,(1,2,3),min_turnover)}


def _v544_forward_outcomes(f,horizon,target_pct):
    highs=f['High'].to_numpy(dtype='float64'); lows=f['Low'].to_numpy(dtype='float64'); closes=f['Close'].to_numpy(dtype='float64')
    hit=np.full(len(f),np.nan); ret=np.full(len(f),np.nan); dd=np.full(len(f),np.nan)
    for i in range(len(f)-horizon):
        hit[i]=1.0 if np.nanmax(highs[i+1:i+1+horizon])>=closes[i]*(1+target_pct) else 0.0
        ret[i]=closes[i+horizon]/closes[i]-1.0
        dd[i]=np.nanmin(lows[i+1:i+1+horizon])/closes[i]-1.0
    return hit,ret,dd


def _v544_compare_static_dynamic_backtest(feat,horizon=5,target_pct=.03,score_threshold=66,min_turnover=0,train_fraction=.70):
    f=feat.copy().dropna(subset=['Close'])
    if len(f)<60:
        return {'static':{},'dynamic':{},'baseline':np.nan,'signal_lift_static':np.nan,'signal_lift_dynamic':np.nan,'train_rows':0,'test_rows':0}
    cut=max(35,min(len(f)-int(horizon)-10,int(len(f)*train_fraction)))
    train=f.iloc[:cut].copy()
    ec=_v544_calibrate_components(train,'early',target_pct,(1,2,3),min_turnover)
    qc=_v544_calibrate_components(train,'quant',target_pct,(1,2,3),min_turnover)
    hit,ret,dd=_v544_forward_outcomes(f,int(horizon),float(target_pct))
    rows=[]
    for i in range(cut,len(f)):
        if not np.isfinite(hit[i]): continue
        r=f.iloc[i]
        ss=score_row(r,min_turnover)[0]
        de,_=_v544_dynamic_score_row(r,ec,'early',min_turnover)
        dq,_=_v544_dynamic_score_row(r,qc,'quant',min_turnover)
        ew=float(np.clip(.35+.10*((ec.get('confidence',0)-qc.get('confidence',0))/100.0),.25,.45))
        pred=(1-ew)*dq+ew*de
        rows.append((ss,pred,hit[i],ret[i],dd[i]))
    if not rows:
        return {'static':{},'dynamic':{},'baseline':np.nan,'signal_lift_static':np.nan,'signal_lift_dynamic':np.nan,'train_rows':len(train),'test_rows':0}
    z=pd.DataFrame(rows,columns=['static','dynamic','hit','ret','dd']); baseline=float(z.hit.mean())
    def pack(col):
        v=z[z[col]>=score_threshold]; n=len(v)
        if not n: return {'n':0,'hits':0,'hit_rate':np.nan,'avg_return':np.nan,'max_drawdown':np.nan,'sample_reliability':0.0}
        return {'n':n,'hits':int(v.hit.sum()),'hit_rate':float(v.hit.mean()),'avg_return':float(v.ret.mean()),'max_drawdown':float(v.dd.min()),'sample_reliability':100*min(1.0,math.sqrt(n/60.0))}
    sb=pack('static'); db=pack('dynamic')
    return {'static':sb,'dynamic':db,'baseline':baseline,'signal_lift_static':(sb.get('hit_rate',np.nan)/baseline if baseline>0 and np.isfinite(sb.get('hit_rate',np.nan)) else np.nan),'signal_lift_dynamic':(db.get('hit_rate',np.nan)/baseline if baseline>0 and np.isfinite(db.get('hit_rate',np.nan)) else np.nan),'train_rows':len(train),'test_rows':len(z)}


def _v544_scan_universe_dynamic(tickers,daily_period='6mo',use_hourly=True,hourly_period='1mo',horizon=5,target_pct=.03,min_turnover=0,buy_threshold=66,prefilter_top=30):
    stage=[]
    for ticker in tickers:
        try:
            d=fetch_ohlcv(ticker,daily_period,'1d')
            if d is None or len(d)<35: continue
            f=compute_features(d); ds=_v544_dynamic_scores(f,target_pct,min_turnover)
            pre=.85*ds['final_prediction']+.15*(.72*ds['static_quant']+.28*ds['static_early'])
            stage.append((pre,ticker,f,ds))
        except Exception:
            continue
    if not stage: return pd.DataFrame()
    stage.sort(key=lambda x:x[0],reverse=True)
    stage=stage[:max(5,min(int(prefilter_top),len(stage)))]
    rows=[]
    for pre,ticker,f,ds in stage:
        try:
            hfeat=m15feat=None; hq=he=np.nan
            if use_hourly:
                h=fetch_ohlcv(ticker,hourly_period,'1h')
                if h is not None and len(h)>=30:
                    hfeat=compute_features(h,True); hr=hfeat.dropna(subset=['Close']).iloc[-1]
                    hq,_=_v544_dynamic_score_row(hr,ds['quant_calibration'],'quant',0)
                    he,_=_v544_dynamic_score_row(hr,ds['early_calibration'],'early',0)
                m15=fetch_ohlcv(ticker,'1mo','15m')
                if m15 is not None and len(m15)>=30: m15feat=compute_features(m15,True)
            dq=ds['dynamic_quant'] if not np.isfinite(hq) else .78*ds['dynamic_quant']+.22*hq
            de=ds['dynamic_early'] if not np.isfinite(he) else .70*ds['dynamic_early']+.30*he
            ew=ds['early_weight']; pred=(1-ew)*dq+ew*de
            precision=m15feat if m15feat is not None else hfeat if hfeat is not None else f
            ent=entry_timing(precision,dq,de)
            cmp=_v544_compare_static_dynamic_backtest(f,horizon,target_pct,buy_threshold,min_turnover)
            db=cmp.get('dynamic',{})
            conf=float(ds['early_calibration'].get('confidence',0)+ds['quant_calibration'].get('confidence',0))/2.0
            lr=f.dropna(subset=['Close']).iloc[-1]
            rows.append({'Ticker':ticker,'Prediction':round(pred,1),'DynamicQuant':round(dq,1),'DynamicEarly':round(de,1),'StaticQuant':round(ds['static_quant'],1),'StaticEarly':round(ds['static_early'],1),'EntryScore':ent['entry_score'],'EntryStatus':ent['status'],'SignalLift':round(cmp.get('signal_lift_dynamic',np.nan),2) if np.isfinite(cmp.get('signal_lift_dynamic',np.nan)) else np.nan,'CalibrationConfidence':round(conf,1),'BacktestN':int(db.get('n',0) or 0),'EmpiricalHitRate':round(float(db.get('hit_rate'))*100,1) if db.get('n',0) and np.isfinite(db.get('hit_rate',np.nan)) else np.nan,'Price':round(float(lr['Close']),4),'VolumeRatio':round(float(lr.get('volume_ratio',np.nan)),2) if np.isfinite(float(lr.get('volume_ratio',np.nan))) else np.nan,'RSI14':round(float(lr.get('rsi14',np.nan)),1) if np.isfinite(float(lr.get('rsi14',np.nan))) else np.nan})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values(['Prediction','DynamicEarly','EntryScore'],ascending=False).reset_index(drop=True) if rows else pd.DataFrame()

# Prefer engine implementations when present; otherwise use the verified local copy.
dynamic_scores = dynamic_scores if callable(dynamic_scores) else _v544_dynamic_scores
compare_static_dynamic_backtest = compare_static_dynamic_backtest if callable(compare_static_dynamic_backtest) else _v544_compare_static_dynamic_backtest
scan_universe_dynamic = scan_universe_dynamic if callable(scan_universe_dynamic) else _v544_scan_universe_dynamic

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


def component_calibration(feat, event_pct=.03, lookbacks=(1,2,3), min_turnover=0):
    """Compare component activation before strong up-days with its normal historical baseline.

    Returns two summary DataFrames (Early and Quant). The event sample is made of the
    1/2/3 trading days before each close-to-close gain >= event_pct. Baseline is all
    eligible historical days. Lift > 1 means the component was active more often before
    strong up-days than on a typical historical day.
    """
    f = feat.copy()
    if f is None or len(f) == 0:
        return pd.DataFrame(), pd.DataFrame(), 0
    f['event_ret_cal'] = f['Close'].pct_change()
    event_indices = [i for i in range(1, len(f)) if pd.notna(f['event_ret_cal'].iloc[i]) and float(f['event_ret_cal'].iloc[i]) >= float(event_pct)]
    if not event_indices:
        return pd.DataFrame(), pd.DataFrame(), 0

    def build(kind):
        # Historical baseline: how often each component is active on an ordinary eligible row.
        baseline = {}
        component_max = {}
        for _, r in f.iterrows():
            comps = early_score_row(r)[1] if kind == 'early' else score_row(r, min_turnover)[1]
            for name, pts, mx in comps:
                component_max[name] = float(mx)
                baseline.setdefault(name, []).append(float(pts) if np.isfinite(float(pts)) else 0.0)

        event_records = []
        event_component_seen = {name: set() for name in component_max}
        by_lb = {lb: {name: [] for name in component_max} for lb in lookbacks}
        for event_no, i in enumerate(event_indices):
            for lb in lookbacks:
                j = i - int(lb)
                if j < 0:
                    continue
                r = f.iloc[j]
                comps = early_score_row(r)[1] if kind == 'early' else score_row(r, min_turnover)[1]
                for name, pts, mx in comps:
                    pts = float(pts) if np.isfinite(float(pts)) else 0.0
                    mx = float(mx)
                    active = pts > 0
                    event_records.append((event_no, lb, name, pts, mx, active))
                    by_lb[lb][name].append(active)
                    if active:
                        event_component_seen.setdefault(name, set()).add(event_no)

        rows=[]
        total_events=len(event_indices)
        for name,mx in component_max.items():
            rec=[x for x in event_records if x[2]==name]
            pts=[x[3] for x in rec]
            active=[x[5] for x in rec]
            base=baseline.get(name,[])
            event_active=100*np.mean(active) if active else np.nan
            base_active=100*np.mean([p>0 for p in base]) if base else np.nan
            lift=(event_active/base_active) if np.isfinite(event_active) and np.isfinite(base_active) and base_active>0 else np.nan
            strength=100*np.mean([p/mx for p in pts]) if pts and mx>0 else np.nan
            coverage=100*len(event_component_seen.get(name,set()))/total_events if total_events else np.nan
            row={
                'Component':name,
                'Event coverage %':coverage,
                'Pre-event active %':event_active,
                'Baseline active %':base_active,
                'Lift x':lift,
                'Avg strength %':strength,
            }
            for lb in sorted(lookbacks, reverse=True):
                vals=by_lb[lb].get(name,[])
                row[f'D-{lb} active %']=100*np.mean(vals) if vals else np.nan
            rows.append(row)
        z=pd.DataFrame(rows)
        if not z.empty:
            z=z.sort_values(['Lift x','Event coverage %','Avg strength %'],ascending=[False,False,False],na_position='last').reset_index(drop=True)
        return z

    return build('early'), build('quant'), len(event_indices)




def normalize_backtest_confidence(bt):
    """V5.3.3 single source of truth for confidence UI and decision logic.

    Recomputes the 80/20 confidence from raw backtest outputs so the displayed
    Backtest Performance, Sample Reliability and Combined Confidence can never
    disagree, even if app.py and quant_engine.py were uploaded out of sync.
    """
    out = dict(bt or {})
    n = int(out.get("n", 0) or 0)
    hr = out.get("hit_rate", np.nan)
    try:
        hr = float(hr)
    except Exception:
        hr = np.nan
    if n <= 0 or not np.isfinite(hr):
        performance = 0.0
        reliability = 0.0
        confidence = 0.0
        label = "LOW"
    else:
        performance = 100.0 * hr
        reliability = 100.0 * min(1.0, math.sqrt(n / 60.0))
        confidence = 0.80 * performance + 0.20 * reliability
        label = "HIGH" if n >= 30 and hr >= 0.60 else ("MEDIUM" if n >= 12 and hr >= 0.45 else "LOW")
    out["backtest_performance"] = round(performance, 1)
    out["sample_reliability"] = round(reliability, 1)
    out["confidence"] = round(confidence, 1)
    out["confidence_label"] = label
    return out

def backtest_display(bt):
    """Avoid presenting a fragile percentage as strong evidence when the sample is tiny."""
    n=int(bt.get('n',0) or 0)
    if n == 0:
        return '—', 'NO SAMPLE'
    hits=int(bt.get('hits', round(float(bt.get('hit_rate',0))*n)))
    if n < 12:
        return f'{hits}/{n} hits', 'LOW SAMPLE'
    return f"{float(bt.get('hit_rate',np.nan))*100:.1f}%", bt.get('confidence_label','')

st.set_page_config(page_title="AI Stock Hunter — V5.7.1",page_icon="📈",layout="wide",initial_sidebar_state="collapsed")
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
st.markdown("""<div class='hero'><span class='badge'>V5.7.1 • WALK-FORWARD VALIDATION • STATIC vs DYNAMIC • SIGNAL LIFT</span><div class='hero-title'>📈 AI Stock Hunter<br>V5.7.1</div><div class='hero-sub'>Smart Money • Early Accumulation • Precision Entry • Walk-forward Backtest • Confidence Engine</div></div>""",unsafe_allow_html=True)

DEFAULT_TICKERS="1196.HK,NVDA,NVMI,MU,AMD,AVGO,AMZN,META,GOOGL,MSFT,AAPL,TSLA,PLTR,CRWV,NBIS,GILD,ORCL,SMCI,ARM,TSM,QCOM,NFLX,UBER,COIN,HOOD,TTWO"
BROAD_200="1196.HK,NVDA,NVMI,MU,AMD,AVGO,AMZN,META,GOOGL,MSFT,AAPL,TSLA,PLTR,CRWV,NBIS,GILD,ORCL,SMCI,ARM,TSM,QCOM,NFLX,UBER,COIN,HOOD,TTWO,ADBE,CRM,INTC,CSCO,AMAT,LRCX,KLAC,MRVL,ADI,TXN,MCHP,ON,MPWR,DELL,HPE,ANET,NET,DDOG,SNOW,ZS,CRWD,PANW,FTNT,OKTA,SHOP,MELI,SE,ABNB,BKNG,DASH,RBLX,SPOT,SNAP,PINS,ROKU,PYPL,XYZ,V,MA,JPM,BAC,WFC,C,GS,MS,AXP,BLK,SCHW,COF,USB,PNC,TFC,BK,STT,BRK-B,UNH,LLY,JNJ,ABBV,MRK,PFE,AMGN,REGN,VRTX,ISRG,MDT,SYK,BSX,EW,TMO,DHR,ABT,MDLZ,KO,PEP,PG,COST,WMT,TGT,HD,LOW,NKE,SBUX,MCD,CMG,YUM,DIS,CMCSA,T,VZ,TMUS,CHTR,XOM,CVX,COP,SLB,EOG,OXY,MPC,VLO,PSX,KMI,WMB,NEE,DUK,SO,AEP,SRE,EXC,D,CEG,VST,CAT,DE,GE,GEV,HON,RTX,LMT,NOC,BA,GD,ETN,EMR,PH,MMM,UPS,FDX,UNP,CSX,NSC,WM,RSG,LIN,APD,SHW,FCX,NEM,NUE,STLD,AA,DOW,DD,GM,F,TM,RIVN,LCID,APO,KKR,BX,ARES,SPGI,MCO,CME,ICE,NDAQ,CB,MMC,AON,PGR,ALL,MET,PRU,AFL,CI,CVS,HUM,CNC,ELV,HCA,IQV,ZTS,BIIB"
VALIDATION_50="NVDA,AMD,MU,AVGO,NVMI,TSM,QCOM,ARM,INTC,AMAT,MSFT,GOOGL,META,AMZN,AAPL,ORCL,CRM,ADBE,NOW,PLTR,JPM,BAC,GS,MS,V,MA,CAT,DE,GE,HON,RTX,BA,LLY,UNH,JNJ,MRK,ABBV,GILD,COST,WMT,KO,PG,MCD,NKE,XOM,CVX,NEE,DIS,UBER,NFLX"
def safe(x,d=2):
    try:return "—" if pd.isna(x) else f"{float(x):.{d}f}"
    except:return "—"
def signal(score,threshold,confidence=None):
    if score>=threshold and confidence!="LOW":return "BUY CANDIDATE"
    if score>=threshold-8:return "WATCH"
    return "AVOID"
def cls(s):return "good" if "BUY" in s or "ENTER" in s else "warn" if "WATCH" in s or "WAIT" in s else "bad"
def chart(df,title):
    # Display-only chart: show the full period selected by the user.
    q=df.copy(); fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.76,.24],vertical_spacing=.04)
    fig.add_trace(go.Candlestick(x=q.index,open=q.Open,high=q.High,low=q.Low,close=q.Close,name="Price"),row=1,col=1)
    if "ema20" in q: fig.add_trace(go.Scatter(x=q.index,y=q.ema20,name="EMA20",line=dict(width=1.4)),row=1,col=1)
    if "ema50" in q: fig.add_trace(go.Scatter(x=q.index,y=q.ema50,name="EMA50",line=dict(width=1.2)),row=1,col=1)
    fig.add_trace(go.Bar(x=q.index,y=q.Volume,name="Volume",opacity=.65),row=2,col=1)
    fig.update_layout(height=570,title=title,template="plotly_dark",paper_bgcolor="#0b0f17",plot_bgcolor="#0b0f17",margin=dict(l=8,r=8,t=45,b=8),xaxis_rangeslider_visible=False,legend_orientation="h")
    return fig

def chart_request(timeframe_label, period_label):
    interval_map={"15m":"15m","1H":"1h","1D":"1d","1W":"1wk"}
    period_map={"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y"}
    interval=interval_map[timeframe_label]
    period=period_map[period_label]
    capped=False
    # Yahoo intraday 15-minute history is limited; cap display history rather than failing.
    if interval=="15m" and period!="1mo":
        period="1mo"
        capped=True
    return interval,period,capped

tab1,tab2,tab3,tab4=st.tabs(["◉ Analyze","⌁ Scanner","▦ Backtest","◎ Validate"])
with tab1:
    c1,c2,c3,c4=st.columns([1.3,1,1,1]); ticker=c1.text_input("Ticker","1196.HK"); hist=c2.selectbox("History",["3mo","6mo","1y"],1); horizon=c3.selectbox("Forecast days",[3,5,7,10],1); target=c4.selectbox("Target %",[3,5,6,8,10,15,20],2)
    buy_threshold=st.slider("BUY threshold",60,80,66)
    gc1,gc2=st.columns(2)
    chart_tf=gc1.selectbox("Chart timeframe",["15m","1H","1D","1W"],2,help="Display only. Does not change Quant, Early, Entry, Calibration or Backtest calculations.")
    chart_period=gc2.selectbox("Chart period",["1M","3M","6M","1Y"],2,help="Display only. 15m history is capped to 1 month by the data provider.")
    st.caption("Chart controls are visual only — model calculations remain unchanged (Quant: Daily + 1H, Early: Daily + 1H, Entry: 15m/1H when available).")
    required_engine_api = ["fetch_ohlcv", "compute_features", "score_latest", "backtest_signal", "entry_timing", "score_row", "early_score_row"]
    missing_engine_api = [name for name in required_engine_api if not hasattr(qe, name)]
    engine_ver = getattr(qe, "ENGINE_VERSION", None)
    engine_build = getattr(qe, "ENGINE_BUILD_ID", None)
    if engine_ver is None and hasattr(qe, "get_engine_version"):
        try:
            engine_ver = qe.get_engine_version()
        except Exception:
            engine_ver = None
    if missing_engine_api:
        st.error("Quant engine is incompatible. Missing: " + ", ".join(missing_engine_api) + ". Upload all 4 files from the V5.5.0 ZIP.")
        st.stop()
    elif engine_ver is not None and str(engine_ver) != APP_VERSION:
        st.warning(f"Version mismatch detected: app V{APP_VERSION} / engine {engine_ver}. Upload all 4 files from the same V{APP_VERSION} ZIP before trusting the results.")
    elif engine_ver is not None:
        st.caption(f"✓ App and quant engine synced: V{APP_VERSION}" + (f" • {engine_build}" if engine_build else ""))
    else:
        st.caption("✓ Quant engine compatibility check passed")
    if st.button("Run V5.5 Dynamic analysis"):
        t=ticker.strip().upper()
        with st.spinner("Running Quant + Early Prediction + Entry Engine..."):
            d=fetch_ohlcv(t,hist,"1d")
            if d is None or len(d)<35: st.error("Not enough market data.")
            else:
                f=compute_features(d); latest=score_latest(f,0)
                dyn=dynamic_scores(f,float(target)/100,0)
                h=fetch_ohlcv(t,"1mo","1h"); hs=he=np.nan
                if h is not None and len(h)>=30:
                    hl=score_latest(compute_features(h,True),0); hs=hl["score"]; he=hl["early_score"]
                qscore=latest["score"] if not np.isfinite(hs) else .78*latest["score"]+.22*hs
                escore=latest["early_score"] if not np.isfinite(he) else .70*latest["early_score"]+.30*he
                bt=backtest_signal(f,int(horizon),float(target)/100,buy_threshold,0)
                bt=normalize_backtest_confidence(bt)
                # 15m precision layer where Yahoo supports it
                m15=fetch_ohlcv(t,"1mo","15m"); ef=compute_features(m15,True) if m15 is not None and len(m15)>=30 else (compute_features(h,True) if h is not None and len(h)>=30 else f)
                ent=entry_timing(ef,qscore,escore); sig=signal(qscore,buy_threshold,bt["confidence_label"])
                st.markdown("<div class='section'>Decision cockpit</div>",unsafe_allow_html=True)
                bt_value, bt_sample_label = backtest_display(bt)
                a,b,c,d1,e=st.columns(5); q_delta=dyn['dynamic_quant']-dyn['static_quant']; e_delta=dyn['dynamic_early']-dyn['static_early']; a.metric("Dynamic Quant",f"{dyn['dynamic_quant']:.1f}",delta=f"{q_delta:+.1f} vs Static {dyn['static_quant']:.1f}"); b.metric("Dynamic Early",f"{dyn['dynamic_early']:.1f}",delta=f"{e_delta:+.1f} vs Static {dyn['static_early']:.1f}"); c.metric("Final Prediction",f"{dyn['final_prediction']:.1f}"); d1.metric("Entry Score",f"{ent['entry_score']:.1f}"); e.metric("Backtest",bt_value)
                if bt['n'] and bt['n'] < 12: st.caption(f"Backtest sample is too small for a reliable percentage: {bt_value}. Treat it as LOW SAMPLE, not as a dependable hit rate.")
                st.markdown(f"<div class='card'><div style='display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap'><div><div class='{cls(sig)}' style='font-size:1.3rem'>{sig}</div><div class='muted'>Confidence: <b>{bt['confidence_label']}</b> ({bt['confidence']:.1f}/100)</div></div><div><div class='{cls(ent['status'])}' style='font-size:1.2rem'>{ent['status']}</div><div class='muted'>Precision Entry Engine • 15m/1h confirmation</div></div></div></div>",unsafe_allow_html=True)
                cp=float(bt.get('backtest_performance',0.0) or 0.0)
                sr=float(bt.get('sample_reliability',0.0) or 0.0)
                cf1,cf2,cf3=st.columns(3)
                cf1.metric("Backtest Performance",f"{cp:.1f}%" if bt['n'] else "—")
                cf2.metric("Sample Reliability",f"{sr:.1f}%" if bt['n'] else "—")
                cf3.metric("Combined Confidence",f"{bt['confidence']:.1f}/100")
                if bt['n']:
                    st.caption(f"Confidence = 80% × Backtest Performance ({cp:.1f}%) + 20% × Sample Reliability ({sr:.1f}%) = {bt['confidence']:.1f}/100. HIGH still requires at least 30 signals and ≥60% hit rate.")
                st.markdown("<div class='section'>Precision entry</div>",unsafe_allow_html=True)
                z1,z2,z3,z4=st.columns(4); z1.metric("Entry zone",f"{ent['zone_low']:.3f} – {ent['zone_high']:.3f}"); z2.metric("Breakout trigger",f"> {ent['trigger']:.3f}"); z3.metric("Invalidation",f"< {ent['invalidation']:.3f}"); z4.metric("Targets",f"{ent['target1']:.3f} / {ent['target2']:.3f}")
                chart_interval,chart_fetch_period,chart_capped=chart_request(chart_tf,chart_period)
                cd=fetch_ohlcv(t,chart_fetch_period,chart_interval)
                if cd is not None and len(cd)>0:
                    cf=compute_features(cd,intraday=chart_interval in ("15m","1h"))
                    st.plotly_chart(chart(cf,f"{t} • {chart_tf} • {chart_period} • Price / EMA / Volume"),use_container_width=True)
                    if chart_capped:
                        st.caption("15m display history is capped to 1M by the data provider. This affects the chart only, not any model calculation.")
                else:
                    st.warning("Selected chart data was unavailable, so the daily analysis chart is shown instead.")
                    st.plotly_chart(chart(f,f"{t} • 1D • Analysis history • Price / EMA / Volume"),use_container_width=True)
                st.markdown("<div class='section'>Dynamic Calibration — current ticker</div>",unsafe_allow_html=True)
                st.caption(f"Final Prediction = {dyn['quant_weight']*100:.0f}% Dynamic Quant + {dyn['early_weight']*100:.0f}% Dynamic Early. Weights are bounded so small samples cannot take over the model.")
                dc1,dc2=st.columns(2)
                with dc1:
                    st.markdown("#### Early dynamic weights")
                    et=dyn['early_calibration']['table'].copy()
                    if not et.empty:
                        for col in ['Base Weight','Dynamic Weight','Lift x','Coverage %','Sample Reliability %','Stability %','Multiplier']:
                            if col in et: et[col]=pd.to_numeric(et[col],errors='coerce').round(2 if col in ['Lift x','Multiplier'] else 1)
                        st.dataframe(et,use_container_width=True,hide_index=True)
                with dc2:
                    st.markdown("#### Quant dynamic weights")
                    qt=dyn['quant_calibration']['table'].copy()
                    if not qt.empty:
                        for col in ['Base Weight','Dynamic Weight','Lift x','Coverage %','Sample Reliability %','Stability %','Multiplier']:
                            if col in qt: qt[col]=pd.to_numeric(qt[col],errors='coerce').round(2 if col in ['Lift x','Multiplier'] else 1)
                        st.dataframe(qt,use_container_width=True,hide_index=True)
                combos=dyn.get('combinations',pd.DataFrame())
                if isinstance(combos,pd.DataFrame) and not combos.empty:
                    st.markdown("#### Strong Early + Quant combinations")
                    cs=combos.copy(); cs['Lift x']=cs['Lift x'].round(2); cs['Coverage %']=cs['Coverage %'].round(1)
                    st.dataframe(cs.head(8),use_container_width=True,hide_index=True)
                st.markdown("<div class='section'>Why this stock?</div>",unsafe_allow_html=True)
                reasons=[]
                # Use the latest feature row for diagnostic fields that are not exported by score_latest().
                lr=f.dropna(subset=['Close']).iloc[-1]
                if float(lr.get('obv_slope5', np.nan))>0: reasons.append("OBV is rising — accumulation pressure")
                if float(lr.get('cmf20', np.nan))>0: reasons.append(f"CMF positive ({float(lr.get('cmf20')):.2f}) — buying flow")
                if bool(lr.get('squeeze', 0)): reasons.append("Bollinger/Keltner squeeze is active")
                if bool(lr.get('squeeze_release', 0)): reasons.append("Squeeze release detected")
                if bool(lr.get('pv_divergence', 0)): reasons.append("Positive price/volume divergence")
                if float(lr.get('roc_accel', np.nan))>0: reasons.append("ROC is accelerating")
                st.markdown("<div class='card'>"+("<br>".join("✓ "+x for x in reasons) if reasons else "No strong early-accumulation confirmation yet.")+"</div>",unsafe_allow_html=True)
                left,right=st.columns(2)
                with left:
                    st.markdown("#### Quant components"); comp=pd.DataFrame(latest['components'],columns=['Component','Points','Max']); comp['Strength %']=(100*comp.Points/comp.Max).round(); st.dataframe(comp,use_container_width=True,hide_index=True)
                with right:
                    st.markdown("#### Early Prediction components"); ec=pd.DataFrame(latest['early_components'],columns=['Component','Points','Max']); ec['Strength %']=(100*ec.Points/ec.Max).round(); st.dataframe(ec,use_container_width=True,hide_index=True)
                st.markdown("<div class='section'>Backtest evidence</div>",unsafe_allow_html=True)
                bt_value, bt_sample_label = backtest_display(bt)
                b1,b2,b3,b4=st.columns(4); b1.metric("Backtest result",bt_value); b2.metric("Sample quality",bt_sample_label); b3.metric("Avg fwd return",f"{bt['avg_return']*100:.2f}%" if bt['n'] else "—"); b4.metric("Max drawdown",f"{bt['max_drawdown']*100:.2f}%" if bt['n'] else "—")
                st.caption(f"A hit means price reached +{target}% within {horizon} trading days after a historical Quant score ≥ {buy_threshold}. For fewer than 12 signals, V5.5.0 shows hits/sample instead of highlighting a fragile percentage.")
                st.markdown("<div class='section'>Static vs Dynamic Backtest</div>",unsafe_allow_html=True)
                cmp=compare_static_dynamic_backtest(f,int(horizon),float(target)/100,buy_threshold,0)
                sb,db=cmp.get('static',{}),cmp.get('dynamic',{})
                def _pct(v): return f"{v*100:.1f}%" if v is not None and np.isfinite(v) else "—"
                def _num(v): return f"{v:.2f}x" if v is not None and np.isfinite(v) else "—"
                compare_df=pd.DataFrame([
                    {'Model':'Static','Signals':sb.get('n',0),'Hit Rate':_pct(sb.get('hit_rate',np.nan)),'Signal Lift':_num(cmp.get('signal_lift_static',np.nan)),'Avg Fwd Return':_pct(sb.get('avg_return',np.nan)),'Max Drawdown':_pct(sb.get('max_drawdown',np.nan)),'Sample Reliability %':round(sb.get('sample_reliability',0),1)},
                    {'Model':'Dynamic','Signals':db.get('n',0),'Hit Rate':_pct(db.get('hit_rate',np.nan)),'Signal Lift':_num(cmp.get('signal_lift_dynamic',np.nan)),'Avg Fwd Return':_pct(db.get('avg_return',np.nan)),'Max Drawdown':_pct(db.get('max_drawdown',np.nan)),'Sample Reliability %':round(db.get('sample_reliability',0),1)}])
                st.dataframe(compare_df,use_container_width=True,hide_index=True)
                st.caption(f"Holdout test: calibration uses the earlier {cmp.get('train_rows',0)} rows and compares Static vs Dynamic on the later {cmp.get('test_rows',0)} eligible rows. Baseline target hit rate in the test period: {_pct(cmp.get('baseline',np.nan))}. Signal Lift = signal hit rate ÷ baseline hit rate.")
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

                    st.markdown("<div class='section'>Early Prediction Calibration Lab</div>",unsafe_allow_html=True)
                    st.caption(f"Uses the same +{target}% event definition. It checks which components were active 1, 2 and 3 trading days before the events, and compares that frequency with their normal historical baseline. Lift above 1.0 is the key diagnostic: the component appeared more often before strong up-days than on a typical day.")
                    early_cal, quant_cal, cal_events = component_calibration(f,float(target)/100,(1,2,3),0)
                    if early_cal.empty:
                        st.info("Not enough events for component calibration in this history window.")
                    else:
                        st.markdown("#### Early components — what actually appeared before the moves")
                        early_show=early_cal.copy()
                        for col in ['Event coverage %','Pre-event active %','Baseline active %','Avg strength %','D-3 active %','D-2 active %','D-1 active %']:
                            if col in early_show: early_show[col]=pd.to_numeric(early_show[col],errors='coerce').round(1)
                        if 'Lift x' in early_show: early_show['Lift x']=pd.to_numeric(early_show['Lift x'],errors='coerce').round(2)
                        st.dataframe(early_show,use_container_width=True,hide_index=True)
                        top=early_show[(early_show['Lift x'].notna()) & (early_show['Event coverage %']>=25)].head(3)
                        if not top.empty:
                            txt=" • ".join([f"{r['Component']}: lift {r['Lift x']:.2f}x, coverage {r['Event coverage %']:.0f}%" for _,r in top.iterrows()])
                            st.success("Best early candidates in this sample: "+txt)
                        st.markdown("#### Quant components — confirmation clues before the moves")
                        quant_show=quant_cal.copy()
                        for col in ['Event coverage %','Pre-event active %','Baseline active %','Avg strength %','D-3 active %','D-2 active %','D-1 active %']:
                            if col in quant_show: quant_show[col]=pd.to_numeric(quant_show[col],errors='coerce').round(1)
                        if 'Lift x' in quant_show: quant_show['Lift x']=pd.to_numeric(quant_show['Lift x'],errors='coerce').round(2)
                        st.dataframe(quant_show,use_container_width=True,hide_index=True)
                        st.caption("Interpretation: Event coverage = share of strong-move events where the component fired at least once in D-3/D-2/D-1. Pre-event active = frequency across all three lookback days. Baseline = normal historical frequency. Lift = pre-event frequency divided by baseline. A high lift with decent coverage is more useful than a high raw activation rate alone.")
with tab2:
    st.markdown("<div class='section'>Multi-stock opportunity scanner</div>",unsafe_allow_html=True)
    universe_mode=st.radio("Universe",["Validation 50","Broad 200","Core 26","Custom"],horizontal=True)
    if universe_mode == "Validation 50":
        default_universe=VALIDATION_50
    elif universe_mode.startswith("Broad"):
        default_universe=BROAD_200
    else:
        default_universe=DEFAULT_TICKERS
    tickers=st.text_area("Tickers",default_universe,disabled=universe_mode!="Custom",height=150)
    c1,c2,c3,c4=st.columns(4); sh=c1.selectbox("History",["3mo","6mo","1y"],1,key="sh"); ho=c2.selectbox("Forecast days",[3,5,7,10],1,key="ho"); ta=c3.selectbox("Target %",[3,5,6,8,10],2,key="ta"); th=c4.number_input("BUY threshold",60,80,66)
    prefilter_default=50 if universe_mode == "Validation 50" else 30
    prefilter_top=st.slider("Deep-analysis finalists",10,50,prefilter_default,5,help="All tickers get the fast daily pass. Only this many finalists get 1h + 15m + entry timing + full backtest. Validation 50 defaults to all 50 so the full basket is ranked.")
    st.caption(f"Stage 1 scans {len([x for x in tickers.split(chr(44)) if x.strip()])} tickers on daily data. Stage 2 performs deep intraday analysis on the top {prefilter_top}. Results are ranked best → worst by Dynamic Prediction, then Dynamic Quant, Dynamic Early and Entry Score.")
    only=st.checkbox("Show only BUY / WATCH",False if universe_mode == "Validation 50" else True)
    if st.button("Scan watchlist"):
        ts=[x.strip().upper() for x in tickers.split(',') if x.strip()]
        with st.spinner("Running V5.7.1 Dynamic Calibration scanner..."): res=scan_universe_dynamic(ts,sh,True,"1mo",int(ho),float(ta)/100,0,int(th),int(prefilter_top))
        if res.empty:st.warning("No valid results returned.")
        else:
            res['Signal']=['BUY CANDIDATE' if r.Prediction>=int(th) and r.CalibrationConfidence>=45 else ('WATCH' if r.Prediction>=int(th)-8 else 'AVOID') for _,r in res.iterrows()]
            res=res.sort_values(['Prediction','DynamicQuant','DynamicEarly','EntryScore'],ascending=[False,False,False,False]).reset_index(drop=True)
            if only:res=res[res.Signal.isin(['BUY CANDIDATE','WATCH'])].reset_index(drop=True)
            res.insert(0,'Rank',range(1,len(res)+1))
            st.session_state.v4res=res; st.session_state.v4time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if 'v4res' in st.session_state:
        res=st.session_state.v4res
        st.caption("Last scan: "+st.session_state.v4time)
        for _,r in res.head(5).iterrows():
            st.markdown(f"<div class='card'><div style='display:flex;justify-content:space-between'><div><b style='font-size:1.3rem'>#{int(r.Rank)} {r.Ticker}</b><div class='{cls(r.Signal)}'>{r.Signal}</div></div><div style='text-align:right'><span class='score'>{r.Prediction:.1f}</span><div class='muted'>Dynamic Prediction</div></div></div><hr style='border-color:#263249'><div class='muted'>Dynamic Early {r.DynamicEarly:.1f} • Dynamic Quant {r.DynamicQuant:.1f} • Lift {safe(r.SignalLift,2)}x • RVOL {safe(r.VolumeRatio)}x • RSI {safe(r.RSI14,1)} • Hit {safe(r.EmpiricalHitRate,1)}% ({int(r.BacktestN)} signals) • Calibration {r.CalibrationConfidence:.0f}%</div></div>",unsafe_allow_html=True)
        cols=['Rank','Ticker','Signal','Prediction','DynamicQuant','DynamicEarly','StaticQuant','StaticEarly','EntryScore','EntryStatus','SignalLift','CalibrationConfidence','Price','VolumeRatio','RSI14','EmpiricalHitRate','BacktestN']
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
                b=backtest_signal(f,int(bt_days),float(bt_target)/100,x,0); rows.append({'Threshold':x,'Hits':b.get('hits',0),'Signals':b['n'],'Hit Rate %':round(b['hit_rate']*100,1) if b['n']>=12 else np.nan,'Sample Quality':'LOW SAMPLE' if 0<b['n']<12 else ('NO SAMPLE' if b['n']==0 else b['confidence_label']),'Avg Return %':round(b['avg_return']*100,2) if b['n'] else np.nan,'Max Drawdown %':round(b['max_drawdown']*100,2) if b['n'] else np.nan,'Confidence Score':b['confidence']})
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
            st.caption("Use this table to choose a threshold from evidence, not from a fixed arbitrary number. For samples below 12 signals, Hit Rate % is intentionally hidden. Prefer a balance of enough signals, hit rate, return and drawdown.")
st.caption("Research prototype. Scores and entry zones are quantitative estimates, not guarantees or personalized investment advice.")


with tab4:
    st.markdown("<div class='section'>V5.7 Walk-Forward Validation Lab</div>", unsafe_allow_html=True)
    st.caption("Four chronological validation folds. Each fold learns only from earlier data and is tested only on the next unseen block. Live model weights are NOT changed.")
    default_universe = VALIDATION_50
    universe_text = st.text_area("Validation universe", default_universe, height=145)
    vc1,vc2,vc3 = st.columns(3)
    vhist = vc1.selectbox("Validation history", ["1y","2y"], 1, key="v_hist")
    vtarget = vc2.selectbox("Validation target %", [3,5,6,8], 0, key="v_target")
    min_events = vc3.selectbox("Minimum train events / fold", [2,3,5,8], 1, key="v_min_events")
    st.caption("Recommended: 50 stocks • 2Y • +3% target • minimum 3 train events/fold. Folds: 40→15%, 55→15%, 70→15%, 85→15%.")

    if st.button("Run V5.7 Walk-Forward Validation", key="run_multi_validation"):
        tickers=[]
        for raw in universe_text.replace("\n",",").split(","):
            t=raw.strip().upper()
            if t and t not in tickers: tickers.append(t)
        tickers=tickers[:50]
        if len(tickers)<3:
            st.error("Enter at least 3 tickers for cross-stock validation.")
        else:
            early_rows=[]; quant_rows=[]; stock_rows=[]; skipped=[]
            bar=st.progress(0, text="Starting V5.7 walk-forward validation...")

            def cal_table(feat,ticker,kind,target,fold,phase):
                cal=_v544_calibrate_components(feat,kind,target,(1,2,3),0)
                tb=cal.get('table',pd.DataFrame()); ev=int(cal.get('events',0) or 0); rows=[]
                if tb is not None and not tb.empty:
                    for _,r in tb.iterrows():
                        rows.append({'Ticker':ticker,'Fold':fold,'Phase':phase,'Component':r['Component'],
                                     'Lift x':float(r.get('Lift x',np.nan)),'Coverage %':float(r.get('Coverage %',np.nan)),
                                     'Events':ev,'Stability %':float(r.get('Stability %',np.nan))})
                return ev,rows

            fold_specs=[(.40,.55),(.55,.70),(.70,.85),(.85,1.00)]
            for idx,t in enumerate(tickers,1):
                bar.progress(int(100*idx/len(tickers)), text=f"Walk-forward {t} ({idx}/{len(tickers)})")
                try:
                    d=fetch_ohlcv(t,vhist,"1d")
                    if d is None: skipped.append((t,"no price data returned")); continue
                    if len(d)<180: skipped.append((t,f"only {len(d)} daily rows; need at least 180")); continue
                    f=compute_features(d).dropna(subset=['Close']).copy()
                    accepted=0; tr_events=0; va_events=0; er=[]; qr=[]
                    for fold,(train_end,test_end) in enumerate(fold_specs,1):
                        i1=max(80,int(len(f)*train_end)); i2=min(len(f),int(len(f)*test_end))
                        if i2-i1<20: continue
                        train=f.iloc[:i1].copy(); test=f.iloc[i1:i2].copy()
                        te,ter=cal_table(train,t,'early',float(vtarget)/100,fold,'Train')
                        tq,tqr=cal_table(train,t,'quant',float(vtarget)/100,fold,'Train')
                        ve,ver=cal_table(test,t,'early',float(vtarget)/100,fold,'Validation')
                        vq,vqr=cal_table(test,t,'quant',float(vtarget)/100,fold,'Validation')
                        tev=min(te,tq); vev=min(ve,vq)
                        if tev<int(min_events) or vev<1: continue
                        accepted+=1; tr_events+=tev; va_events+=vev; er.extend(ter+ver); qr.extend(tqr+vqr)
                    if accepted<2: skipped.append((t,f"only {accepted} usable walk-forward folds")); continue
                    early_rows.extend(er); quant_rows.extend(qr)
                    stock_rows.append({'Ticker':t,'Rows':len(f),'Usable folds':accepted,'Train events*':tr_events,'Validation events':va_events})
                except Exception as e:
                    skipped.append((t,f"calculation/data error: {type(e).__name__}"))
            bar.empty()

            def aggregate_walkforward(rows):
                z=pd.DataFrame(rows)
                if z.empty:return z
                out=[]
                for comp,g in z.groupby('Component'):
                    train=g[g.Phase=='Train']; val=g[g.Phase=='Validation']
                    keys=set(zip(train.Ticker,train.Fold)).intersection(set(zip(val.Ticker,val.Fold)))
                    tr=[]; va=[]; cov=[]; stab=[]; stock_pos={}; fold_pos={i:[] for i in range(1,5)}; events=0
                    for ticker,fold in keys:
                        ar=train[(train.Ticker==ticker)&(train.Fold==fold)]; br=val[(val.Ticker==ticker)&(val.Fold==fold)]
                        if ar.empty or br.empty: continue
                        tl=float(ar['Lift x'].iloc[0]); vl=float(br['Lift x'].iloc[0])
                        if np.isfinite(tl):tr.append(tl)
                        if np.isfinite(vl):
                            va.append(vl); stock_pos.setdefault(ticker,[]).append(vl>1.0); fold_pos[fold].append(vl>1.0)
                        cv=float(br['Coverage %'].iloc[0]); sv=float(br['Stability %'].iloc[0])
                        if np.isfinite(cv):cov.append(cv)
                        if np.isfinite(sv):stab.append(sv)
                        events+=int(br['Events'].iloc[0] or 0)
                    if not va: continue
                    tmed=float(np.median(tr)) if tr else np.nan; vmed=float(np.median(va)); vmean=float(np.mean(va))
                    pstock=sum(1 for vals in stock_pos.values() if np.mean(vals)>0.5); ns=len(stock_pos); spr=100*pstock/ns if ns else 0
                    pfold=sum(1 for vals in fold_pos.values() if vals and np.mean(vals)>0.5); nf=sum(1 for vals in fold_pos.values() if vals); fpr=100*pfold/nf if nf else 0
                    repeat=np.isfinite(tmed) and tmed>1.0 and vmed>1.0
                    if ns>=8 and nf>=3 and repeat and vmed>=1.05 and spr>=60 and fpr>=75: verdict='ROBUST WF WINNER'
                    elif ns>=8 and repeat and spr>=50: verdict='PROMISING'
                    elif ns>=8 and np.isfinite(tmed) and tmed>1.05 and (vmed<0.95 or spr<=40): verdict='FAILED WF'
                    else: verdict='MIXED / WEAK'
                    out.append({'Component':comp,'Train Median Lift':tmed,'WF Median Lift':vmed,'WF Mean Lift':vmean,
                                'WF Coverage %':float(np.mean(cov)) if cov else np.nan,'Positive stocks':f'{pstock}/{ns}',
                                'Positive stocks %':spr,'Positive folds':f'{pfold}/{nf}','Positive folds %':fpr,
                                'WF Stability %':float(np.mean(stab)) if stab else np.nan,'Validation events*':events,'Verdict':verdict})
                out=pd.DataFrame(out)
                if out.empty:return out
                rank={'ROBUST WF WINNER':0,'PROMISING':1,'MIXED / WEAK':2,'FAILED WF':3}
                out['_r']=out.Verdict.map(rank).fillna(9)
                return out.sort_values(['_r','WF Median Lift','Positive stocks %','Positive folds %'],ascending=[True,False,False,False]).drop(columns='_r').reset_index(drop=True)

            ea=aggregate_walkforward(early_rows); qa=aggregate_walkforward(quant_rows)
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Stocks validated",f"{len(stock_rows)}/{len(tickers)}"); m2.metric("Skipped",len(skipped))
            m3.metric("Walk-forward folds","4"); m4.metric("Validation events*",sum(r['Validation events'] for r in stock_rows))
            st.caption("*Training event totals can repeat across expanding windows. Validation blocks are chronological and non-overlapping.")

            for title,z in [("Early — walk-forward validation",ea),("Quant — walk-forward validation",qa)]:
                st.markdown(f"<div class='section'>{title}</div>",unsafe_allow_html=True)
                if z.empty: st.warning("Not enough walk-forward evidence for this section.")
                else:
                    show=z.copy()
                    for c in ['Train Median Lift','WF Median Lift','WF Mean Lift','WF Coverage %','Positive stocks %','Positive folds %','WF Stability %']:
                        show[c]=pd.to_numeric(show[c],errors='coerce').round(2)
                    st.dataframe(show,use_container_width=True,hide_index=True)
                    winners=z[z.Verdict=='ROBUST WF WINNER']
                    if not winners.empty: st.success("Robust walk-forward winners: "+', '.join(winners.Component.head(5).tolist()))
                    failed=z[z.Verdict=='FAILED WF']
                    if not failed.empty: st.warning("Strong in training but failed walk-forward: "+', '.join(failed.Component.head(5).tolist()))

            st.markdown("<div class='section'>Stock / fold audit</div>",unsafe_allow_html=True)
            if stock_rows: st.dataframe(pd.DataFrame(stock_rows),use_container_width=True,hide_index=True)
            if skipped:
                with st.expander(f"Skipped stocks ({len(skipped)})"):
                    st.dataframe(pd.DataFrame(skipped,columns=['Ticker','Reason']),use_container_width=True,hide_index=True)
            st.info("V5.7 gate: do not change live weights yet. A component should survive multiple unseen chronological folds and many stocks before V5.8 weight optimization.")

    st.markdown("<div class='section'>V5.7 Liquidity fix</div>",unsafe_allow_html=True)
    st.caption("Liquidity now compares current dollar turnover with the stock's rolling 60-bar median: <0.8x = 0, 0.8–1.2x = 2, 1.2–1.5x = 4, ≥1.5x = 6. A non-zero minimum-turnover setting remains a separate tradability floor.")

