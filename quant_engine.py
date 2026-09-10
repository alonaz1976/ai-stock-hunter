from __future__ import annotations
import math
import numpy as np
import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True,
                         progress=False, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(-1):
                try: df = df.xs(ticker, axis=1, level=-1)
                except Exception: pass
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in needed): return None
        return df[needed].dropna().copy()
    except Exception:
        return None


def ema(s, span): return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(close, n=14):
    d = close.diff(); gain = d.clip(lower=0); loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100/(1+rs)

def true_range(df):
    p = df["Close"].shift(1)
    return pd.concat([(df.High-df.Low), (df.High-p).abs(), (df.Low-p).abs()], axis=1).max(axis=1)

def adx(df, n=14):
    up=df.High.diff(); down=-df.Low.diff()
    pdm=pd.Series(np.where((up>down)&(up>0),up,0.), index=df.index)
    mdm=pd.Series(np.where((down>up)&(down>0),down,0.), index=df.index)
    atr=true_range(df).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    pdi=100*pdm.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/atr
    mdi=100*mdm.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def _clamp(v, lo=0, hi=100): return max(lo, min(hi, v))

def compute_features(df: pd.DataFrame, intraday=False):
    x=df.copy(); c=x.Close; v=x.Volume.astype(float)
    x["ret1"]=c.pct_change(); x["mom3"]=c.pct_change(3); x["mom5"]=c.pct_change(5); x["mom10"]=c.pct_change(10)
    x["ema9"]=ema(c,9); x["ema20"]=ema(c,20); x["ema50"]=ema(c,50)
    vm20=v.rolling(20).mean(); vs20=v.rolling(20).std()
    x["volume_ratio"]=v/vm20; x["volume_z"]=(v-vm20)/vs20.replace(0,np.nan); x["vol_accel"]=v.rolling(3).mean()/v.rolling(10).mean()
    x["rsi14"]=rsi(c,14)
    macd=ema(c,12)-ema(c,26); sig=ema(macd,9)
    x["macd"]=macd; x["macd_signal"]=sig; x["macd_hist"]=macd-sig; x["macd_hist_slope"]=x.macd_hist.diff(2)
    tr=true_range(x); x["atr14"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean(); x["atr_pct"]=x.atr14/c*100; x["adx14"]=adx(x,14)
    ph=x.High.rolling(20).max().shift(1); pl=x.Low.rolling(20).min().shift(1)
    x["breakout20_pct"]=(c/ph-1)*100; x["dist_to_high20_pct"]=(ph/c-1)*100
    x["support20"]=pl; x["resistance20"]=ph
    rng=(x.High-x.Low).replace(0,np.nan); x["close_location"]=(c-x.Low)/rng; x["turnover"]=c*v
    # OBV + A/D + CMF
    direction=np.sign(c.diff()).fillna(0); x["obv"]=(direction*v).cumsum(); x["obv_slope5"]=x.obv.diff(5)/(v.rolling(20).mean()*5).replace(0,np.nan)
    mfm=((c-x.Low)-(x.High-c))/rng; mfv=mfm.fillna(0)*v
    x["ad_line"]=mfv.cumsum(); x["ad_slope5"]=x.ad_line.diff(5)/(v.rolling(20).mean()*5).replace(0,np.nan)
    x["cmf20"]=mfv.rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)
    # Bollinger / Keltner squeeze
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); bb_u=mid+2*sd; bb_l=mid-2*sd
    kc_u=x.ema20+1.5*x.atr14; kc_l=x.ema20-1.5*x.atr14
    x["bb_width"]=(bb_u-bb_l)/mid.replace(0,np.nan); x["squeeze"]=(bb_u<kc_u)&(bb_l>kc_l)
    x["squeeze_release"]=(~x.squeeze)&x.squeeze.shift(1).fillna(False)
    # Relative-strength proxy versus own 20D baseline + ROC acceleration
    x["roc5"]=c.pct_change(5); x["roc10"]=c.pct_change(10); x["roc_accel"]=x.roc5-(x.roc10/2)
    x["rs20"]=c/c.shift(20)-1
    # Price/volume divergence: flat/soft price while OBV improves
    x["pv_divergence"]=(c.pct_change(5)<=0.02)&(x.obv.diff(5)>0)
    # Session VWAP for intraday; rolling VWAP proxy for daily
    typical=(x.High+x.Low+x.Close)/3
    if intraday and isinstance(x.index, pd.DatetimeIndex):
        dates=pd.Series(x.index.date,index=x.index)
        x["vwap"]=(typical*v).groupby(dates).cumsum()/v.groupby(dates).cumsum().replace(0,np.nan)
    else:
        x["vwap"]=(typical*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)
    return x


def score_row(r, min_turnover=3_000_000):
    comp=[]
    m3=float(r.get("mom3",np.nan)); m5=float(r.get("mom5",np.nan)); s=0
    if np.isfinite(m3) and np.isfinite(m5):
        s += 5 if m3>0 else 0; s += 4 if m5>0 else 0; s += 3 if m3>.02 else 0; s += 2 if m5>.04 else 0; s += 2 if 0<m3<.12 else 0; s -= 4 if m3>.20 else 0
    s=_clamp(s,0,16); comp.append(("Momentum",s,16))
    vr=float(r.get("volume_ratio",np.nan)); vz=float(r.get("volume_z",np.nan)); va=float(r.get("vol_accel",np.nan)); s=0
    if np.isfinite(vr): s+=(5 if vr>=1.2 else 0)+(4 if vr>=1.5 else 0)+(3 if vr>=2 else 0)
    if np.isfinite(vz) and vz>=1:s+=4
    if np.isfinite(va) and va>=1.15:s+=4
    s=_clamp(s,0,20); comp.append(("Volume acceleration",s,20))
    b=float(r.get("breakout20_pct",np.nan)); s=0
    if np.isfinite(b):
        if -3<=b<=0:s=9
        elif 0<b<=4:s=14
        elif 4<b<=8:s=10
        elif -6<=b<-3:s=5
    comp.append(("20D breakout/proximity",s,14))
    rv=float(r.get("rsi14",np.nan)); s=0
    if np.isfinite(rv):
        if 52<=rv<=67:s=8
        elif 48<=rv<52 or 67<rv<=72:s=5
        elif 40<=rv<48:s=2
    comp.append(("RSI",s,8))
    mh=float(r.get("macd_hist",np.nan)); ms=float(r.get("macd_hist_slope",np.nan)); s=0
    if np.isfinite(mh) and mh>0:s+=7
    if np.isfinite(ms) and ms>0:s+=5
    comp.append(("MACD histogram",s,12))
    av=float(r.get("adx14",np.nan)); s=10 if np.isfinite(av) and av>=30 else 8 if np.isfinite(av) and av>=25 else 5 if np.isfinite(av) and av>=20 else 0; comp.append(("ADX trend strength",s,10))
    at=float(r.get("atr_pct",np.nan)); s=8 if np.isfinite(at) and 2<=at<=7 else 5 if np.isfinite(at) and (1.2<=at<2 or 7<at<=10) else 1 if np.isfinite(at) and at>12 else 0; comp.append(("ATR / volatility",s,8))
    cl=float(r.get("close_location",np.nan)); s=6 if np.isfinite(cl) and cl>=.75 else 4 if np.isfinite(cl) and cl>=.55 else 2 if np.isfinite(cl) and cl>=.4 else 0; comp.append(("Candle strength",s,6))
    tv=float(r.get("turnover",np.nan)); s=6 if np.isfinite(tv) and tv>=min_turnover*3 else 4 if np.isfinite(tv) and tv>=min_turnover else 2 if np.isfinite(tv) and tv>=min_turnover*.5 else 0; comp.append(("Liquidity",s,6))
    return float(_clamp(sum(p for _,p,_ in comp))),comp


def early_score_row(r):
    pts=[]
    ob=float(r.get("obv_slope5",np.nan)); s=18 if np.isfinite(ob) and ob>.08 else 12 if np.isfinite(ob) and ob>0 else 0; pts.append(("OBV accumulation",s,18))
    cm=float(r.get("cmf20",np.nan)); s=18 if np.isfinite(cm) and cm>.12 else 12 if np.isfinite(cm) and cm>0 else 0; pts.append(("CMF money flow",s,18))
    ad=float(r.get("ad_slope5",np.nan)); s=12 if np.isfinite(ad) and ad>.05 else 8 if np.isfinite(ad) and ad>0 else 0; pts.append(("Accumulation/Distribution",s,12))
    sq=bool(r.get("squeeze",False)); rel=bool(r.get("squeeze_release",False)); s=14 if rel else 9 if sq else 0; pts.append(("Bollinger/Keltner squeeze",s,14))
    rs=float(r.get("rs20",np.nan)); s=12 if np.isfinite(rs) and rs>.08 else 8 if np.isfinite(rs) and rs>0 else 0; pts.append(("Relative strength",s,12))
    ra=float(r.get("roc_accel",np.nan)); s=10 if np.isfinite(ra) and ra>.02 else 6 if np.isfinite(ra) and ra>0 else 0; pts.append(("ROC acceleration",s,10))
    div=bool(r.get("pv_divergence",False)); pts.append(("Price/Volume divergence",10 if div else 0,10))
    vr=float(r.get("volume_ratio",np.nan)); va=float(r.get("vol_accel",np.nan)); s=6 if (np.isfinite(vr) and .8<=vr<1.5 and np.isfinite(va) and va>1) else 3 if np.isfinite(va) and va>1 else 0; pts.append(("Early RVOL",s,6))
    return float(_clamp(sum(p for _,p,_ in pts))),pts


def score_latest(feat, min_turnover=3_000_000):
    r=feat.dropna(subset=["Close"]).iloc[-1]; score,components=score_row(r,min_turnover); early,early_components=early_score_row(r)
    keys=["volume_ratio","volume_z","rsi14","adx14","atr_pct","breakout20_pct","cmf20","obv_slope5","ad_slope5","rs20","roc_accel","vwap","ema9","ema20","ema50","support20","resistance20","macd_hist"]
    out={"score":score,"early_score":early,"price":float(r.Close),"daily_change_pct":float(r.get("ret1",np.nan))*100,"turnover":float(r.get("turnover",np.nan)),"components":components,"early_components":early_components}
    for k in keys: out[k]=float(r.get(k,np.nan)) if not pd.isna(r.get(k,np.nan)) else np.nan
    out["squeeze"]=bool(r.get("squeeze",False)); out["squeeze_release"]=bool(r.get("squeeze_release",False)); out["pv_divergence"]=bool(r.get("pv_divergence",False))
    return out


def backtest_signal(feat, horizon=5, target_pct=.06, score_threshold=66, min_turnover=3_000_000):
    f=feat.copy(); f["score"]=[score_row(r,min_turnover)[0] for _,r in f.iterrows()]
    highs=f.High.to_numpy(); lows=f.Low.to_numpy(); closes=f.Close.to_numpy(); hits=np.full(len(f),np.nan); rets=np.full(len(f),np.nan); dds=np.full(len(f),np.nan)
    for i in range(len(f)-horizon):
        fh=highs[i+1:i+1+horizon]; fl=lows[i+1:i+1+horizon]; end=closes[i+horizon]
        hits[i]=1. if np.nanmax(fh)>=closes[i]*(1+target_pct) else 0.
        rets[i]=end/closes[i]-1; dds[i]=np.nanmin(fl)/closes[i]-1
    f["hit"]=hits; f["fwd_return"]=rets; f["drawdown"]=dds
    valid=f[(f.score>=score_threshold)&f.hit.notna()].copy(); n=len(valid)
    if n==0:return {"n":0,"hit_rate":np.nan,"avg_return":np.nan,"max_drawdown":np.nan,"confidence":0,"confidence_label":"LOW"}
    hr=float(valid.hit.mean()); sample_factor=min(1.,math.sqrt(n/60)); conf=100*(.65*hr+.35*sample_factor); label="HIGH" if n>=30 and hr>=.60 else "MEDIUM" if n>=12 and hr>=.45 else "LOW"
    return {"n":int(n),"hit_rate":hr,"avg_return":float(valid.fwd_return.mean()),"max_drawdown":float(valid.drawdown.min()),"confidence":round(conf,1),"confidence_label":label}


def entry_timing(feat, quant_score, early_score):
    r=feat.dropna(subset=["Close"]).iloc[-1]; p=float(r.Close); atr=float(r.get("atr14",np.nan)); vwap=float(r.get("vwap",np.nan)); e9=float(r.get("ema9",np.nan)); e20=float(r.get("ema20",np.nan)); res=float(r.get("resistance20",np.nan)); sup=float(r.get("support20",np.nan)); vr=float(r.get("volume_ratio",np.nan)); mh=float(r.get("macd_hist",np.nan)); rv=float(r.get("rsi14",np.nan))
    score=0
    if np.isfinite(vwap) and p>=vwap:score+=20
    if np.isfinite(e9) and np.isfinite(e20) and e9>=e20:score+=18
    if np.isfinite(vr) and vr>=1.15:score+=16
    if np.isfinite(mh) and mh>0:score+=14
    if np.isfinite(rv) and 50<=rv<=70:score+=12
    score += min(20,max(0,(quant_score-55)*.8)); score=float(_clamp(score))
    a=atr if np.isfinite(atr) and atr>0 else p*.02
    zone_low=max(x for x in [p-.35*a, vwap if np.isfinite(vwap) else p-.35*a, e9 if np.isfinite(e9) else p-.35*a] if np.isfinite(x))
    zone_high=p+.10*a; trigger=max(p, res if np.isfinite(res) and res<p+1.2*a else p)+.08*a
    invalid=max(sup if np.isfinite(sup) else p-1.5*a, p-1.6*a)
    stretched=np.isfinite(e20) and p>e20+1.5*a
    near_break=np.isfinite(res) and 0 <= (res-p)/p <= .015
    if stretched: status="WAIT FOR PULLBACK"
    elif near_break and (not np.isfinite(vr) or vr<1.2): status="WAIT FOR BREAKOUT"
    elif score>=72 and quant_score>=66: status="ENTER ZONE"
    else: status="WATCH ENTRY"
    return {"entry_score":round(score,1),"status":status,"zone_low":zone_low,"zone_high":zone_high,"trigger":trigger,"invalidation":invalidation,"target1":p*1.03,"target2":p*1.06}


def scan_universe(tickers,daily_period="6mo",use_hourly=True,hourly_period="1mo",horizon=5,target_pct=.06,min_turnover=3_000_000,buy_threshold=66,prefilter_top=30):
    """Two-stage scanner.

    Stage 1: cheap daily features for the full universe and keep only the strongest
    candidates. Stage 2: hourly + 15m confirmation, entry timing and backtest only
    for the finalists. This makes 100-200 ticker scans much more practical.
    """
    stage1=[]
    for ticker in tickers:
        d=fetch_ohlcv(ticker,daily_period,"1d")
        if d is None or len(d)<35: continue
        f=compute_features(d); latest=score_latest(f,min_turnover)
        # Daily pre-score intentionally rewards both current setup and early accumulation.
        pre=.72*latest["score"]+.28*latest["early_score"]
        stage1.append((pre,ticker,d,f,latest))
    if not stage1: return pd.DataFrame()
    stage1.sort(key=lambda z:z[0],reverse=True)
    finalists=stage1[:max(5,min(int(prefilter_top),len(stage1)))]

    rows=[]
    for pre,ticker,d,f,latest in finalists:
        hs=he=np.nan; hfeat=None; m15feat=None
        if use_hourly:
            h=fetch_ohlcv(ticker,hourly_period,"1h")
            if h is not None and len(h)>=30:
                hfeat=compute_features(h,True); hl=score_latest(hfeat,0); hs=hl["score"]; he=hl["early_score"]
            # 15m is fetched only for finalists, not the entire universe.
            m15=fetch_ohlcv(ticker,"1mo","15m")
            if m15 is not None and len(m15)>=30:
                m15feat=compute_features(m15,True)
        final=latest["score"] if not np.isfinite(hs) else .78*latest["score"]+.22*hs
        early=latest["early_score"] if not np.isfinite(he) else .70*latest["early_score"]+.30*he
        precision_feat=m15feat if m15feat is not None else hfeat if hfeat is not None else f
        ent=entry_timing(precision_feat,final,early)
        bt=backtest_signal(f,horizon,target_pct,buy_threshold,min_turnover)
        rows.append({"Ticker":ticker,"Score":round(final,1),"EarlyScore":round(early,1),"PreScore":round(pre,1),"EntryScore":ent["entry_score"],"EntryStatus":ent["status"],"EntryLow":round(ent["zone_low"],4),"EntryHigh":round(ent["zone_high"],4),"Trigger":round(ent["trigger"],4),"Invalidation":round(ent["invalidation"],4),"Price":round(latest["price"],4),"DailyChangePct":round(latest["daily_change_pct"],2),"VolumeRatio":round(latest["volume_ratio"],2) if np.isfinite(latest["volume_ratio"]) else np.nan,"RSI14":round(latest["rsi14"],1) if np.isfinite(latest["rsi14"]) else np.nan,"ADX14":round(latest["adx14"],1) if np.isfinite(latest["adx14"]) else np.nan,"ATRpct":round(latest["atr_pct"],2) if np.isfinite(latest["atr_pct"]) else np.nan,"CMF20":round(latest["cmf20"],3) if np.isfinite(latest["cmf20"]) else np.nan,"HourlyScore":round(hs,1) if np.isfinite(hs) else np.nan,"EmpiricalHitRate":round(bt["hit_rate"]*100,1) if bt["n"] else np.nan,"BacktestN":bt["n"],"Confidence":bt["confidence_label"],"ConfidenceScore":bt["confidence"],"AvgReturn":round(bt["avg_return"]*100,2) if bt["n"] else np.nan,"MaxDrawdown":round(bt["max_drawdown"]*100,2) if bt["n"] else np.nan})
    return pd.DataFrame(rows).sort_values(["Score","EarlyScore","EntryScore"],ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
