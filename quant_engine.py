from __future__ import annotations
import math
import numpy as np
import pandas as pd
import yfinance as yf

ENGINE_VERSION = "5.9.8"
ENGINE_BUILD_ID = "V597-REAL-PREMARKET-20260915-A"

def get_engine_version():
    return ENGINE_VERSION


def _series1d(obj, index=None, name=None):
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            idx = index if index is not None else obj.index
            return pd.Series(np.full(len(idx), np.nan), index=idx, name=name, dtype='float64')
        obj = obj.iloc[:, 0]
    if isinstance(obj, pd.Series):
        idx = obj.index if index is None else index
        vals = pd.to_numeric(obj, errors='coerce').to_numpy(dtype='float64', na_value=np.nan)
        return pd.Series(vals.reshape(-1), index=idx, name=name, dtype='float64')
    arr = np.asarray(obj, dtype='float64').reshape(-1)
    if index is None or len(index) != len(arr):
        index = pd.RangeIndex(len(arr))
    return pd.Series(arr, index=index, name=name, dtype='float64')


def _canonical_ohlcv(df: pd.DataFrame, ticker: str | None = None):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    cols = list(df.columns)
    ticker_u = str(ticker).upper() if ticker else None

    def parts(col):
        return [str(x) for x in col] if isinstance(col, tuple) else [str(col)]

    chosen = {}
    for field in required:
        matches = []
        for pos, col in enumerate(cols):
            pp = parts(col)
            if any(x.lower() == field.lower() for x in pp):
                ticker_match = bool(ticker_u and any(x.upper() == ticker_u for x in pp))
                matches.append((0 if ticker_match else 1, pos))
        if not matches:
            return None
        matches.sort()
        chosen[field] = matches[0][1]

    out = pd.DataFrame(index=df.index.copy())
    for field in required:
        s = _series1d(df.iloc[:, chosen[field]], index=df.index, name=field)
        out[field] = s.to_numpy(dtype='float64')
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=required)
    return out if not out.empty else None


def fetch_ohlcv(ticker: str, period='6mo', interval='1d'):
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by='column',
        )
        return _canonical_ohlcv(raw, ticker=ticker)
    except Exception:
        return None



def fetch_premarket_snapshots(tickers):
    """Return real US pre-market snapshots from Yahoo intraday extended-hours data.

    The volume-strength ratio compares today's cumulative pre-market volume with
    prior available pre-market sessions through the same clock time.  Non-US
    tickers are intentionally left to the UI as N/A rather than fabricating an
    auction/pre-open equivalent.
    """
    from datetime import datetime, time as dtime
    from zoneinfo import ZoneInfo

    names=[str(t).upper().strip() for t in tickers if str(t).strip()]
    if not names:
        return {}
    out={}
    try:
        raw=yf.download(names, period='5d', interval='5m', prepost=True,
                        auto_adjust=False, progress=False, threads=True, group_by='ticker')
    except Exception:
        return out
    if raw is None or raw.empty:
        return out

    now=datetime.now(ZoneInfo('America/New_York'))
    today=now.date(); cutoff=now.time().replace(tzinfo=None)
    for tkr in names:
        try:
            if len(names)==1:
                q=raw.copy()
            elif isinstance(raw.columns,pd.MultiIndex):
                if tkr in raw.columns.get_level_values(0): q=raw[tkr].copy()
                elif tkr in raw.columns.get_level_values(-1): q=raw.xs(tkr,axis=1,level=-1).copy()
                else: continue
            else:
                continue
            if q.empty: continue
            idx=pd.DatetimeIndex(q.index)
            if idx.tz is None: idx=idx.tz_localize('UTC')
            idx=idx.tz_convert('America/New_York'); q.index=idx
            close=pd.to_numeric(q.get('Close'),errors='coerce'); vol=pd.to_numeric(q.get('Volume'),errors='coerce').fillna(0)
            dates=pd.Series(q.index.date,index=q.index)
            times=pd.Series(q.index.time,index=q.index)
            pm_mask=(dates==today) & (times>=dtime(4,0)) & (times<dtime(9,30)) & (times<=cutoff)
            pm=q.loc[pm_mask]
            if pm.empty: continue
            pm_close=pd.to_numeric(pm['Close'],errors='coerce').dropna()
            if pm_close.empty: continue
            last_price=float(pm_close.iloc[-1]); pm_volume=float(pd.to_numeric(pm['Volume'],errors='coerce').fillna(0).sum())
            regular=(times>=dtime(9,30)) & (times<dtime(16,0)) & (dates<today)
            prev=q.loc[regular]
            if prev.empty: continue
            prev_dates=sorted(set(prev.index.date))
            prev_day=prev_dates[-1]; prev_day_rows=prev[prev.index.date==prev_day]
            prev_close_series=pd.to_numeric(prev_day_rows['Close'],errors='coerce').dropna()
            if prev_close_series.empty: continue
            prev_close=float(prev_close_series.iloc[-1])
            change=(last_price/prev_close-1.0)*100.0 if prev_close else np.nan

            hist=[]
            for d in sorted(set(q.index.date)):
                if d>=today: continue
                m=(q.index.date==d) & (q.index.time>=dtime(4,0)) & (q.index.time<dtime(9,30)) & (q.index.time<=cutoff)
                v=float(pd.to_numeric(q.loc[m,'Volume'],errors='coerce').fillna(0).sum())
                if v>0: hist.append(v)
            baseline=float(np.median(hist)) if hist else np.nan
            strength=(pm_volume/baseline) if baseline and np.isfinite(baseline) and baseline>0 else np.nan
            out[tkr]={'PMPrice':last_price,'PMChangePct':change,'PMVolume':pm_volume,'PMVolumeStrength':strength,'PMData':'REAL 5M'}
        except Exception:
            continue
    return out


def fetch_aftermarket_snapshots(tickers):
    """Return real US after-hours snapshots from Yahoo 5-minute extended-hours data.

    AH volume strength compares cumulative after-hours volume through the current
    clock time with prior available sessions through the same elapsed AH window.
    """
    from datetime import datetime, time as dtime
    from zoneinfo import ZoneInfo
    names=[str(t).upper().strip() for t in tickers if str(t).strip()]
    if not names: return {}
    out={}
    try:
        raw=yf.download(names, period='5d', interval='5m', prepost=True,
                        auto_adjust=False, progress=False, threads=True, group_by='ticker')
    except Exception:
        return out
    if raw is None or raw.empty: return out
    now=datetime.now(ZoneInfo('America/New_York')); today=now.date(); cutoff=now.time().replace(tzinfo=None)
    for tkr in names:
        try:
            if len(names)==1: q=raw.copy()
            elif isinstance(raw.columns,pd.MultiIndex):
                if tkr in raw.columns.get_level_values(0): q=raw[tkr].copy()
                elif tkr in raw.columns.get_level_values(-1): q=raw.xs(tkr,axis=1,level=-1).copy()
                else: continue
            else: continue
            if q.empty: continue
            idx=pd.DatetimeIndex(q.index)
            if idx.tz is None: idx=idx.tz_localize('UTC')
            idx=idx.tz_convert('America/New_York'); q.index=idx
            ah=q[(q.index.date==today) & (q.index.time>=dtime(16,0)) & (q.index.time<dtime(20,0)) & (q.index.time<=cutoff)]
            if ah.empty: continue
            ah_close=pd.to_numeric(ah['Close'],errors='coerce').dropna()
            if ah_close.empty: continue
            last_price=float(ah_close.iloc[-1]); ah_volume=float(pd.to_numeric(ah['Volume'],errors='coerce').fillna(0).sum())
            reg=q[(q.index.date==today) & (q.index.time>=dtime(9,30)) & (q.index.time<dtime(16,0))]
            if reg.empty: continue
            reg_close=pd.to_numeric(reg['Close'],errors='coerce').dropna()
            if reg_close.empty: continue
            close=float(reg_close.iloc[-1]); change=(last_price/close-1.0)*100.0 if close else np.nan
            hist=[]
            for d in sorted(set(q.index.date)):
                if d>=today: continue
                m=(q.index.date==d) & (q.index.time>=dtime(16,0)) & (q.index.time<dtime(20,0)) & (q.index.time<=cutoff)
                v=float(pd.to_numeric(q.loc[m,'Volume'],errors='coerce').fillna(0).sum())
                if v>0: hist.append(v)
            baseline=float(np.median(hist)) if hist else np.nan
            strength=(ah_volume/baseline) if baseline and np.isfinite(baseline) and baseline>0 else np.nan
            out[tkr]={'AHPrice':last_price,'AHChangePct':change,'AHVolume':ah_volume,'AHVolumeStrength':strength,'AHData':'REAL 5M'}
        except Exception: continue
    return out


def ema(s, span):
    s = _series1d(s)
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close, n=14):
    c = _series1d(close, name='Close')
    d = c.diff()
    gain = d.clip(lower=0.0)
    loss = -d.clip(upper=0.0)
    ag = gain.ewm(alpha=1.0/float(n), adjust=False, min_periods=int(n)).mean()
    al = loss.ewm(alpha=1.0/float(n), adjust=False, min_periods=int(n)).mean()
    num = ag.to_numpy(dtype='float64')
    den = al.to_numpy(dtype='float64')
    rs = np.full(len(num), np.nan, dtype='float64')
    np.divide(num, den, out=rs, where=np.isfinite(den) & (den != 0.0))
    out = 100.0 - (100.0 / (1.0 + rs))
    out[(den == 0.0) & np.isfinite(num) & (num > 0.0)] = 100.0
    return pd.Series(out, index=c.index, dtype='float64')


def true_range(df):
    high = _series1d(df['High'], index=df.index)
    low = _series1d(df['Low'], index=df.index)
    close = _series1d(df['Close'], index=df.index)
    prev = close.shift(1)
    arr = np.vstack([
        (high-low).to_numpy(),
        (high-prev).abs().to_numpy(),
        (low-prev).abs().to_numpy(),
    ])
    return pd.Series(np.nanmax(arr, axis=0), index=df.index, dtype='float64')


def _wilder(series, n):
    s = _series1d(series)
    return s.ewm(alpha=1.0/float(n), adjust=False, min_periods=int(n)).mean()


def adx(df, n=14):
    high = _series1d(df['High'], index=df.index)
    low = _series1d(df['Low'], index=df.index)
    up = high.diff()
    down = -low.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index, dtype='float64')
    mdm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index, dtype='float64')
    atr = _wilder(true_range(df), n)
    atr_arr = atr.to_numpy(dtype='float64')
    pnum = 100 * _wilder(pdm, n).to_numpy(dtype='float64')
    mnum = 100 * _wilder(mdm, n).to_numpy(dtype='float64')
    pdi = np.full(len(atr), np.nan)
    mdi = np.full(len(atr), np.nan)
    np.divide(pnum, atr_arr, out=pdi, where=np.isfinite(atr_arr) & (atr_arr != 0))
    np.divide(mnum, atr_arr, out=mdi, where=np.isfinite(atr_arr) & (atr_arr != 0))
    den = pdi + mdi
    dx = np.full(len(den), np.nan)
    np.divide(100*np.abs(pdi-mdi), den, out=dx, where=np.isfinite(den) & (den != 0))
    return _wilder(pd.Series(dx, index=df.index), n)


def _clamp(v, lo=0, hi=100):
    return max(lo, min(hi, v))


def compute_features(df: pd.DataFrame, intraday=False):
    x = _canonical_ohlcv(df)
    if x is None or len(x) == 0:
        raise ValueError('No valid OHLCV rows')
    c = _series1d(x['Close'], index=x.index)
    v = _series1d(x['Volume'], index=x.index)

    x['ret1'] = c.pct_change()
    x['mom3'] = c.pct_change(3)
    x['mom5'] = c.pct_change(5)
    x['mom10'] = c.pct_change(10)
    x['ema9'] = ema(c, 9)
    x['ema20'] = ema(c, 20)
    x['ema50'] = ema(c, 50)

    vm20 = v.rolling(20).mean()
    vs20 = v.rolling(20).std()
    x['volume_ratio'] = v / vm20.replace(0, np.nan)
    x['volume_z'] = (v-vm20) / vs20.replace(0, np.nan)
    x['vol_accel'] = v.rolling(3).mean() / v.rolling(10).mean().replace(0, np.nan)
    # V5.9 robust/transition volume features. Median-based RVOL is intentionally
    # less sensitive to one-off extreme sessions than the ordinary rolling mean.
    vmed20 = v.rolling(20, min_periods=10).median()
    x['robust_volume_ratio'] = v / vmed20.replace(0, np.nan)
    x['volume_reexpansion'] = x['robust_volume_ratio'] / x['robust_volume_ratio'].shift(1).replace(0, np.nan)
    x['volume_dryup'] = (1.0 - v / v.rolling(5, min_periods=2).max().replace(0, np.nan)).clip(0, 1)

    x['rsi14'] = rsi(c, 14)
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    x['macd'] = macd
    x['macd_signal'] = sig
    x['macd_hist'] = macd - sig
    x['macd_hist_slope'] = x['macd_hist'].diff(2)
    x['ema9_cross_up'] = ((x['ema9'] > x['ema20']) & (x['ema9'].shift(1) <= x['ema20'].shift(1))).fillna(False).astype('int8')
    x['macd_cross_up'] = ((x['macd'] > x['macd_signal']) & (x['macd'].shift(1) <= x['macd_signal'].shift(1))).fillna(False).astype('int8')
    x['macd_hist_turn_pos'] = ((x['macd_hist'] > 0) & (x['macd_hist'].shift(1) <= 0)).fillna(False).astype('int8')

    tr = true_range(x)
    x['atr14'] = _wilder(tr, 14)
    x['atr_pct'] = x['atr14'] / c * 100
    x['adx14'] = adx(x, 14)

    ph = x['High'].rolling(20).max().shift(1)
    pl = x['Low'].rolling(20).min().shift(1)
    x['breakout20_pct'] = (c/ph - 1) * 100
    x['support20'] = pl
    x['resistance20'] = ph

    rng = (x['High'] - x['Low']).replace(0, np.nan)
    x['close_location'] = (c - x['Low']) / rng
    x['turnover'] = c * v
    x['turnover_median60'] = x['turnover'].rolling(60, min_periods=20).median()
    x['turnover_ratio60'] = x['turnover'] / x['turnover_median60'].replace(0, np.nan)

    direction = np.sign(c.diff()).fillna(0)
    x['obv'] = (direction * v).cumsum()
    basev = (v.rolling(20).mean() * 5).replace(0, np.nan)
    x['obv_slope5'] = x['obv'].diff(5) / basev

    mfm = ((c-x['Low']) - (x['High']-c)) / rng
    mfv = mfm.fillna(0) * v
    x['ad_line'] = mfv.cumsum()
    x['ad_slope5'] = x['ad_line'].diff(5) / basev
    x['cmf20'] = mfv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    bb_u = mid + 2*sd
    bb_l = mid - 2*sd
    kc_u = x['ema20'] + 1.5*x['atr14']
    kc_l = x['ema20'] - 1.5*x['atr14']
    sq = ((bb_u < kc_u) & (bb_l > kc_l)).fillna(False).astype('int8')
    x['squeeze'] = sq
    x['squeeze_release'] = ((sq.shift(1) == 1) & (sq == 0)).fillna(False).astype('int8')

    x['roc5'] = c.pct_change(5)
    x['roc10'] = c.pct_change(10)
    x['roc_accel'] = x['roc5'] - x['roc10']/2
    x['rs20'] = c/c.shift(20) - 1
    x['pv_divergence'] = ((c.pct_change(5) <= 0.02) & (x['obv'].diff(5) > 0)).fillna(False).astype('int8')

    typical = (x['High'] + x['Low'] + x['Close']) / 3
    if intraday and isinstance(x.index, pd.DatetimeIndex):
        dates = pd.Series(x.index.date, index=x.index)
        x['vwap'] = (typical*v).groupby(dates).cumsum() / v.groupby(dates).cumsum().replace(0, np.nan)
    else:
        x['vwap'] = (typical*v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    x['vwap_cross_up'] = ((c > x['vwap']) & (c.shift(1) <= x['vwap'].shift(1))).fillna(False).astype('int8')
    # Impulse -> hold -> dry-up -> re-expansion sequence features. All are causal.
    x['impulse_1d'] = x['ret1']
    x['impulse_3d'] = c.pct_change(3)
    x['recent_impulse'] = x['ret1'].rolling(5, min_periods=1).max()
    recent_close_high = c.rolling(5, min_periods=1).max()
    x['post_impulse_retention'] = (c / recent_close_high.replace(0, np.nan)).clip(0, 1.05)
    x['fresh_transition_count'] = (x['ema9_cross_up'] + x['macd_cross_up'] + x['macd_hist_turn_pos'] + x['vwap_cross_up']).astype('float64')
    return x



def explosive_score_row(r):
    """V5.9 Explosive Move fingerprint. Independent from Quant/Entry scores."""
    comps=[]
    imp=max(float(r.get('impulse_1d', np.nan)) if np.isfinite(float(r.get('impulse_1d', np.nan))) else -1,
            float(r.get('impulse_3d', np.nan))/2 if np.isfinite(float(r.get('impulse_3d', np.nan))) else -1)
    p=18 if imp>=.08 else 14 if imp>=.045 else 9 if imp>=.025 else 4 if imp>0 else 0
    comps.append(('Impulse strength',p,18))

    rr=float(r.get('robust_volume_ratio',np.nan)); vz=float(r.get('volume_z',np.nan))
    p=0
    if np.isfinite(rr): p=18 if rr>=3 else 15 if rr>=2 else 11 if rr>=1.5 else 6 if rr>=1.15 else 0
    if np.isfinite(vz) and vz>=2: p=min(18,p+3)
    comps.append(('Robust volume shock',p,18))

    ret=float(r.get('post_impulse_retention',np.nan)); ri=float(r.get('recent_impulse',np.nan))
    p=14 if np.isfinite(ret) and np.isfinite(ri) and ri>=.035 and ret>=.97 else 10 if np.isfinite(ret) and np.isfinite(ri) and ri>=.025 and ret>=.94 else 4 if np.isfinite(ret) and ret>=.92 else 0
    comps.append(('Post-impulse retention',p,14))

    dry=float(r.get('volume_dryup',np.nan))
    p=10 if np.isfinite(dry) and dry>=.55 and np.isfinite(ret) and ret>=.94 else 7 if np.isfinite(dry) and dry>=.35 and np.isfinite(ret) and ret>=.92 else 0
    comps.append(('Volume dry-up',p,10))

    re=float(r.get('volume_reexpansion',np.nan))
    p=10 if np.isfinite(re) and re>=1.8 else 7 if np.isfinite(re) and re>=1.35 else 3 if np.isfinite(re) and re>=1.1 else 0
    comps.append(('Volume re-expansion',p,10))

    tr=float(r.get('fresh_transition_count',np.nan))
    mh=float(r.get('macd_hist_slope',np.nan)); va=float(r.get('vol_accel',np.nan))
    p=min(12,(4*int(tr) if np.isfinite(tr) else 0) + (2 if np.isfinite(mh) and mh>0 else 0) + (2 if np.isfinite(va) and va>1.1 else 0))
    comps.append(('Fresh transitions',p,12))

    atr=float(r.get('atr_pct',np.nan))
    p=8 if np.isfinite(atr) and atr>=6 else 6 if np.isfinite(atr) and atr>=4 else 4 if np.isfinite(atr) and atr>=2.5 else 1 if np.isfinite(atr) else 0
    comps.append(('ATR move capacity',p,8))

    e9=float(r.get('ema9',np.nan)); e20=float(r.get('ema20',np.nan)); c=float(r.get('Close',np.nan)); vw=float(r.get('vwap',np.nan)); rs=float(r.get('rsi14',np.nan))
    p=(4 if np.isfinite(e9) and np.isfinite(e20) and e9>=e20 else 0)+(4 if np.isfinite(c) and np.isfinite(vw) and c>=vw else 0)+(2 if np.isfinite(rs) and 48<=rs<=75 else 0)
    comps.append(('Trend/VWAP structure',p,10))

    score=float(_clamp(sum(x[1] for x in comps)))
    exhausted=(np.isfinite(rs) and rs>80) or (np.isfinite(float(r.get('mom5',np.nan))) and float(r.get('mom5',0))>.28)
    if exhausted and score<82: stage='EXHAUSTED'
    elif score>=82 and p>=6: stage='TRIGGERED'
    elif score>=68: stage='ARMED'
    elif score>=52: stage='BUILDING'
    elif score>=35: stage='RESET'
    else: stage='COLD'
    return score, comps, stage


def hourly_confirmation(feat):
    if feat is None or len(feat)==0:return {'score':np.nan,'status':'NO DATA'}
    r=feat.dropna(subset=['Close']).iloc[-1]
    score=0.0
    c=float(r.get('Close',np.nan)); vw=float(r.get('vwap',np.nan)); e9=float(r.get('ema9',np.nan)); e20=float(r.get('ema20',np.nan)); mh=float(r.get('macd_hist',np.nan)); rs=float(r.get('rsi14',np.nan)); rr=float(r.get('robust_volume_ratio',np.nan)); tr=float(r.get('fresh_transition_count',0) or 0)
    if np.isfinite(c) and np.isfinite(vw) and c>=vw:score+=20
    if np.isfinite(e9) and np.isfinite(e20) and e9>=e20:score+=20
    if np.isfinite(mh) and mh>0:score+=16
    if np.isfinite(rs) and 50<=rs<=72:score+=14
    if np.isfinite(rr) and rr>=1.2:score+=14
    score+=min(16,4*tr)
    score=float(_clamp(score))
    return {'score':round(score,1),'status':'CONFIRMED' if score>=70 else 'PARTIAL' if score>=50 else 'WEAK'}


def explosive_probabilities(feat, current_score=None, min_sample=8):
    """Empirical, shrinkage-calibrated probabilities from prior causal rows only."""
    f=feat.copy().dropna(subset=['Close'])
    if len(f)<35:return {'sample':0,'threshold':np.nan,'probs':{}}
    scores=np.array([explosive_score_row(r)[0] for _,r in f.iterrows()],dtype=float)
    cs=float(scores[-1] if current_score is None else current_score)
    threshold=max(45.0, cs-10.0)
    out={}
    highs=f['High'].to_numpy(dtype=float); closes=f['Close'].to_numpy(dtype=float)
    for h in (1,3,5):
        for target in (.05,.10,.15):
            vals=[]; base=[]
            for i in range(0,len(f)-h-1):
                hit=1.0 if np.nanmax(highs[i+1:i+1+h])>=closes[i]*(1+target) else 0.0
                base.append(hit)
                if scores[i]>=threshold: vals.append(hit)
            n=len(vals); b=float(np.mean(base)) if base else np.nan
            if n:
                # mild prior toward the unconditional base rate; prevents tiny buckets from reading 0/100%.
                p=(float(np.sum(vals))+4.0*b)/(n+4.0) if np.isfinite(b) else float(np.mean(vals))
            else:p=np.nan
            out[f'p{int(target*100)}_{h}d']={'prob':100*p if np.isfinite(p) else np.nan,'n':n,'baseline':100*b if np.isfinite(b) else np.nan,'reliable':n>=min_sample}
    return {'sample':int(sum(1 for x in scores[:-6] if x>=threshold)),'threshold':round(threshold,1),'probs':out}


def explosive_latest(feat, hourly_feat=None):
    r=feat.dropna(subset=['Close']).iloc[-1]
    score, comps, stage=explosive_score_row(r)
    pr=explosive_probabilities(feat,score)
    hc=hourly_confirmation(hourly_feat) if hourly_feat is not None else {'score':np.nan,'status':'NO DATA'}
    out={'score':round(score,1),'stage':stage,'components':comps,'hourly_confirmation':hc['score'],'hourly_status':hc['status'],'prob_threshold':pr.get('threshold',np.nan),'prob_sample':pr.get('sample',0)}
    for k,v in pr.get('probs',{}).items():
        out[k]=v.get('prob',np.nan); out[k+'_n']=v.get('n',0); out[k+'_reliable']=v.get('reliable',False)
    for k in ['robust_volume_ratio','post_impulse_retention','volume_dryup','volume_reexpansion','fresh_transition_count','atr_pct']:
        val=r.get(k,np.nan); out[k]=float(val) if pd.notna(val) else np.nan
    return out


def explosive_walkforward(feat, threshold=68):
    """Chronological four-fold validation of the static V5.9 fingerprint."""
    f=feat.copy().dropna(subset=['Close'])
    if len(f)<120:return pd.DataFrame()
    scores=np.array([explosive_score_row(r)[0] for _,r in f.iterrows()],dtype=float)
    highs=f['High'].to_numpy(dtype=float); closes=f['Close'].to_numpy(dtype=float)
    rows=[]; specs=[(.40,.55),(.55,.70),(.70,.85),(.85,1.00)]
    for target in (.05,.10,.15):
        for h in (1,3,5):
            fold_lifts=[]; fold_hits=[]; fold_ns=[]
            for fold,(a,b) in enumerate(specs,1):
                i1=int(len(f)*a); i2=min(int(len(f)*b),len(f)-h-1)
                sig=[]; base=[]
                for i in range(i1,i2):
                    hit=1.0 if np.nanmax(highs[i+1:i+1+h])>=closes[i]*(1+target) else 0.0
                    base.append(hit)
                    if scores[i]>=threshold:sig.append(hit)
                n=len(sig); bhr=float(np.mean(base)) if base else np.nan; hr=float(np.mean(sig)) if sig else np.nan
                lift=hr/bhr if np.isfinite(hr) and np.isfinite(bhr) and bhr>0 else np.nan
                if np.isfinite(lift):fold_lifts.append(lift)
                if np.isfinite(hr):fold_hits.append(hr)
                fold_ns.append(n)
            pos=sum(1 for x in fold_lifts if x>1.0)
            rows.append({'Target':f'+{int(target*100)}%','Horizon':f'{h}D','Signals':sum(fold_ns),'Hit Rate %':100*np.average(fold_hits,weights=[max(1,n) for n in fold_ns if n>0]) if fold_hits and len(fold_hits)==sum(1 for n in fold_ns if n>0) else (100*np.mean(fold_hits) if fold_hits else np.nan),'Median Lift':float(np.median(fold_lifts)) if fold_lifts else np.nan,'Positive Folds':f'{pos}/{len(fold_lifts)}'})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# V5.9.2 Adaptive Target/Horizon Research Layer
# Strictly causal features: scores at row t use row t and earlier only. Forward
# highs are used only as labels after the score has been frozen.
# -----------------------------------------------------------------------------
def move_score_row(r):
    """Short-horizon 3-5% move score, deliberately separate from Explosive Score."""
    pts=[]
    rr=float(r.get('robust_volume_ratio',np.nan)); va=float(r.get('vol_accel',np.nan))
    p=22 if np.isfinite(rr) and rr>=2 else 17 if np.isfinite(rr) and rr>=1.5 else 10 if np.isfinite(rr) and rr>=1.15 else 0
    if np.isfinite(va) and va>=1.25:p=min(25,p+3)
    pts.append(('Volume acceleration',p,25))
    m3=float(r.get('mom3',np.nan)); m5=float(r.get('mom5',np.nan)); p=0
    if np.isfinite(m3): p += 12 if .01<=m3<=.08 else 7 if m3>0 else 0
    if np.isfinite(m5): p += 8 if .015<=m5<=.12 else 4 if m5>0 else 0
    pts.append(('Near-term momentum',min(20,p),20))
    tr=float(r.get('fresh_transition_count',0) or 0); mh=float(r.get('macd_hist_slope',np.nan))
    p=min(20,5*tr + (5 if np.isfinite(mh) and mh>0 else 0)); pts.append(('Fresh transitions',p,20))
    c=float(r.get('Close',np.nan)); vw=float(r.get('vwap',np.nan)); e9=float(r.get('ema9',np.nan)); e20=float(r.get('ema20',np.nan))
    p=(8 if np.isfinite(c) and np.isfinite(vw) and c>=vw else 0)+(7 if np.isfinite(e9) and np.isfinite(e20) and e9>=e20 else 0); pts.append(('VWAP/EMA structure',p,15))
    rs=float(r.get('rsi14',np.nan)); p=10 if np.isfinite(rs) and 50<=rs<=72 else 5 if np.isfinite(rs) and 45<=rs<=78 else 0; pts.append(('RSI window',p,10))
    atr=float(r.get('atr_pct',np.nan)); p=10 if np.isfinite(atr) and atr>=3 else 7 if np.isfinite(atr) and atr>=2 else 3 if np.isfinite(atr) else 0; pts.append(('Move capacity',p,10))
    return float(_clamp(sum(x[1] for x in pts))),pts

def _entry_proxy_row(r, quant=None):
    c=float(r.get('Close',np.nan)); vw=float(r.get('vwap',np.nan)); e9=float(r.get('ema9',np.nan)); e20=float(r.get('ema20',np.nan)); rr=float(r.get('robust_volume_ratio',r.get('volume_ratio',np.nan))); mh=float(r.get('macd_hist',np.nan)); rs=float(r.get('rsi14',np.nan))
    s=0
    if np.isfinite(c) and np.isfinite(vw) and c>=vw:s+=20
    if np.isfinite(e9) and np.isfinite(e20) and e9>=e20:s+=18
    if np.isfinite(rr) and rr>=1.15:s+=16
    if np.isfinite(mh) and mh>0:s+=14
    if np.isfinite(rs) and 50<=rs<=70:s+=12
    if quant is not None and np.isfinite(quant):s+=min(20,max(0,float(quant)*.20))
    return float(_clamp(s))

def historical_signal_timeline(feat):
    f=feat.copy().dropna(subset=['Close'])
    if f.empty:return pd.DataFrame()
    highs=f['High'].to_numpy(float); closes=f['Close'].to_numpy(float)
    rows=[]
    for i,(idx,r) in enumerate(f.iterrows()):
        ex,_,stage=explosive_score_row(r); mv,_=move_score_row(r); q,_=score_row(r,0); en=_entry_proxy_row(r,q)
        row={'Date':idx,'Close':closes[i],'Move Score':round(mv,1),'Explosive Score':round(ex,1),'Entry Score':round(en,1),'Stage':stage,'Volume Ratio':r.get('robust_volume_ratio',np.nan),'ATR %':r.get('atr_pct',np.nan)}
        for h in (1,3,5):
            if i+h < len(f):
                row[f'+{h}D Close Return %']=100*(closes[i+h]/closes[i]-1)
                row[f'+{h}D Max Return %']=100*(np.nanmax(highs[i+1:i+1+h])/closes[i]-1)
            else:
                row[f'+{h}D Close Return %']=np.nan; row[f'+{h}D Max Return %']=np.nan
        rows.append(row)
    out=pd.DataFrame(rows)
    # Signal Timing V5.9.4: causal score acceleration. Positive means the score is strengthening now.
    for score in ("Move Score","Explosive Score","Entry Score"):
        for lag in (1,2,3):
            out[f"{score} Accel {lag}D"] = pd.to_numeric(out[score],errors="coerce") - pd.to_numeric(out[score],errors="coerce").shift(lag)
    ex=pd.to_numeric(out["Explosive Score"],errors="coerce")
    out["Explosive Rising 3D"] = ((ex.diff()>0)&(ex.diff().shift(1)>0)&(ex.diff().shift(2)>0))
    out["Signal Acceleration"] = out["Explosive Score Accel 3D"]
    return out

def signal_timing_latest(feat, hourly_feat=None):
    """Current timing layer: score trajectory + hourly confirmation + evidence-based stage."""
    tl=historical_signal_timeline(feat)
    if tl.empty:return {}
    r=tl.iloc[-1]
    hc=hourly_confirmation(hourly_feat) if hourly_feat is not None and len(hourly_feat) else {"score":np.nan,"status":"NO DATA"}
    ex=float(r.get("Explosive Score",np.nan)); mv=float(r.get("Move Score",np.nan)); en=float(r.get("Entry Score",np.nan))
    a1=float(r.get("Explosive Score Accel 1D",np.nan)); a2=float(r.get("Explosive Score Accel 2D",np.nan)); a3=float(r.get("Explosive Score Accel 3D",np.nan))
    rising=bool(r.get("Explosive Rising 3D",False)); hcs=float(hc.get("score",np.nan))
    # Conservative operational labels; research sheets validate these rather than treating them as probabilities.
    if max(ex,mv) < 50: stage="WAIT"
    elif max(ex,mv) < 60 or (np.isfinite(a3) and a3<=0): stage="WATCH"
    elif max(ex,mv)>=60 and en>=55 and (rising or (np.isfinite(a3) and a3>=6)): stage="ARMED"
    else: stage="WATCH"
    if max(ex,mv)>=68 and en>=62 and np.isfinite(a3) and a3>=8 and np.isfinite(hcs) and hcs>=60:
        stage="TRIGGER"
    return {"move_score":round(mv,1),"explosive_score":round(ex,1),"entry_score":round(en,1),
            "accel_1d":round(a1,1) if np.isfinite(a1) else np.nan,"accel_2d":round(a2,1) if np.isfinite(a2) else np.nan,"accel_3d":round(a3,1) if np.isfinite(a3) else np.nan,
            "rising_3d":rising,"hourly_confirmation":round(hcs,1) if np.isfinite(hcs) else np.nan,"hourly_status":hc.get("status","NO DATA"),"timing_stage":stage}

def acceleration_validation(timeline, targets=(3,5,10,15), horizons=(1,3,5), accel_thresholds=(0,3,5,8,10,15)):
    """Validate whether score acceleration adds lift; future returns are labels only."""
    if timeline is None or timeline.empty:return pd.DataFrame()
    t=timeline.copy(); rows=[]
    for window in (1,2,3):
      acol=f"Explosive Score Accel {window}D"
      if acol not in t:continue
      a=pd.to_numeric(t[acol],errors="coerce")
      for th in accel_thresholds:
       mask=a>=th
       for target in targets:
        for h in horizons:
         col=f"+{h}D Max Return %"
         if col not in t:continue
         y=pd.to_numeric(t[col],errors="coerce")>=target; valid=pd.to_numeric(t[col],errors="coerce").notna() & a.notna()
         n=int((mask&valid).sum()); base=float(y[valid].mean()) if valid.any() else np.nan; hit=float(y[mask&valid].mean()) if n else np.nan
         lift=hit/base if np.isfinite(hit) and np.isfinite(base) and base>0 else np.nan
         rows.append({"Window":f"{window}D","Accel Threshold":th,"Target":f"+{target}%","Horizon":f"{h}D","Signals":n,"Hit Rate %":100*hit if np.isfinite(hit) else np.nan,"Baseline %":100*base if np.isfinite(base) else np.nan,"Lift":lift})
    return pd.DataFrame(rows)

def threshold_optimization(feat, thresholds=(50,55,60,65,70,75,80), targets=(.03,.05,.10,.15), horizons=(1,3,5), score_kind='explosive'):
    f=feat.copy().dropna(subset=['Close'])
    if len(f)<120:return pd.DataFrame()
    scorer=(lambda r: move_score_row(r)[0]) if score_kind=='move' else (lambda r: explosive_score_row(r)[0])
    scores=np.array([scorer(r) for _,r in f.iterrows()]); highs=f['High'].to_numpy(float); closes=f['Close'].to_numpy(float)
    specs=[(.40,.55),(.55,.70),(.70,.85),(.85,1.00)]; rows=[]
    for th in thresholds:
      for target in targets:
       for h in horizons:
        lifts=[]; ns=[]; hrs=[]; fwds=[]
        for a,b in specs:
          i1=int(len(f)*a); i2=min(int(len(f)*b),len(f)-h)
          base=[]; sig=[]; rets=[]
          for i in range(i1,i2):
            mx=np.nanmax(highs[i+1:i+1+h])/closes[i]-1; hit=float(mx>=target); base.append(hit)
            if scores[i]>=th:sig.append(hit); rets.append(mx)
          if sig:
            hr=float(np.mean(sig)); br=float(np.mean(base)) if base else np.nan; lift=hr/br if br>0 else np.nan
            if np.isfinite(lift):lifts.append(lift)
            hrs.extend(sig); fwds.extend(rets); ns.append(len(sig))
        rows.append({'Score':score_kind.title(),'Threshold':th,'Target':f'+{int(target*100)}%','Horizon':f'{h}D','Signals':sum(ns),'Hit Rate %':100*np.mean(hrs) if hrs else np.nan,'Median Lift':np.median(lifts) if lifts else np.nan,'Positive Folds':f'{sum(x>1 for x in lifts)}/{len(lifts)}','Avg Max Forward %':100*np.mean(fwds) if fwds else np.nan})
    return pd.DataFrame(rows)

def adaptive_target_horizon(opt, min_signals=20, min_positive_fold_ratio=.5):
    if opt is None or opt.empty:return {}
    x=opt.copy(); x['FoldRatio']=x['Positive Folds'].astype(str).apply(lambda z:(float(z.split('/')[0])/float(z.split('/')[1])) if '/' in z and float(z.split('/')[1]) else 0)
    x=x[(x['Signals']>=min_signals)&(x['FoldRatio']>=min_positive_fold_ratio)&pd.notna(x['Median Lift'])]
    if x.empty:return {}
    x=x.assign(RankScore=x['Median Lift']*np.sqrt(x['Signals'].clip(lower=1))*x['FoldRatio'])
    r=x.sort_values(['RankScore','Median Lift'],ascending=False).iloc[0]
    return {'Score':r['Score'],'Threshold':int(r['Threshold']),'Target':r['Target'],'Horizon':r['Horizon'],'Signals':int(r['Signals']),'Median Lift':float(r['Median Lift']),'Positive Folds':r['Positive Folds'],'Avg Max Forward %':float(r['Avg Max Forward %'])}



def reliability_score_row(r, min_signals=20):
    """0-100 research reliability: lift + sample + fold consistency, with explicit penalties for thin folds."""
    try:
        lift=float(r.get('Median Lift', np.nan)); n=int(r.get('Signals',0)); pf=str(r.get('Positive Folds','0/0'))
        a,b=pf.split('/'); pos=float(a); folds=float(b); fr=pos/folds if folds else 0.0
    except Exception:
        return 0.0
    if not np.isfinite(lift) or n<=0 or folds<=0:return 0.0
    lift_component=35*min(1.0,max(0.0,(lift-1.0)/2.0))
    sample_component=25*min(1.0,math.sqrt(n/max(1.0,float(min_signals)*4.0)))
    fold_component=25*fr
    coverage_component=15*min(1.0,folds/4.0)
    penalty=0.55 if folds<2 else (0.82 if folds<3 else 1.0)
    return round(_clamp((lift_component+sample_component+fold_component+coverage_component)*penalty),1)

def adaptive_target_horizon_v2(opt, min_signals=20, min_positive_fold_ratio=.5, min_folds=2, min_reliability=45):
    if opt is None or opt.empty:return {}
    x=opt.copy()
    def parts(z):
        try:
            a,b=str(z).split('/'); return float(a),float(b)
        except:return 0.0,0.0
    pp=x['Positive Folds'].apply(parts); x['PositiveFoldCount']=[z[0] for z in pp]; x['FoldCount']=[z[1] for z in pp]
    x['FoldRatio']=x['PositiveFoldCount']/x['FoldCount'].replace(0,np.nan)
    x['Reliability Score']=x.apply(lambda r: reliability_score_row(r,min_signals),axis=1)
    x=x[(x['Signals']>=min_signals)&(x['FoldCount']>=min_folds)&(x['FoldRatio']>=min_positive_fold_ratio)&(x['Reliability Score']>=min_reliability)&pd.notna(x['Median Lift'])]
    if x.empty:return {}
    # Reliability dominates; lift and sample break ties without allowing a 1-fold anomaly to win.
    x=x.assign(RankScore=x['Reliability Score'] + 8*np.log1p(x['Median Lift'].clip(lower=1)) + 2*np.log1p(x['Signals']))
    r=x.sort_values(['RankScore','Reliability Score','Median Lift','Signals'],ascending=False).iloc[0]
    return {'Score':r['Score'],'Threshold':int(r['Threshold']),'Target':r['Target'],'Horizon':r['Horizon'],'Signals':int(r['Signals']),'Median Lift':float(r['Median Lift']),'Positive Folds':r['Positive Folds'],'Reliability Score':float(r['Reliability Score']),'Avg Max Forward %':float(r['Avg Max Forward %'])}

def pre_move_study(timeline, targets=(5,10,15), horizons=(1,3,5)):
    if timeline is None or timeline.empty:return pd.DataFrame()
    t=timeline.reset_index(drop=True).copy(); rows=[]
    for target in targets:
      for h in horizons:
        col=f'+{h}D Max Return %'
        if col not in t:continue
        events=t.index[pd.to_numeric(t[col],errors='coerce')>=target].tolist()
        for lag in (3,2,1,0):
          ix=[i-lag for i in events if i-lag>=0]
          if not ix:continue
          z=t.loc[ix]
          rows.append({'Target':f'+{target}%','Horizon':f'{h}D','Days Before':lag,'Events':len(z),'Move Score Avg':pd.to_numeric(z['Move Score'],errors='coerce').mean(),'Explosive Score Avg':pd.to_numeric(z['Explosive Score'],errors='coerce').mean(),'Entry Score Avg':pd.to_numeric(z['Entry Score'],errors='coerce').mean(),'Volume Ratio Avg':pd.to_numeric(z['Volume Ratio'],errors='coerce').mean(),'ATR % Avg':pd.to_numeric(z['ATR %'],errors='coerce').mean()})
    return pd.DataFrame(rows)

def topk_daily_validation(timeline_all, k_values=(5,10), horizons=(1,3,5), score_col='Explosive Score'):
    if timeline_all is None or timeline_all.empty or 'Ticker' not in timeline_all:return pd.DataFrame()
    x=timeline_all.copy(); x['Date']=pd.to_datetime(x['Date']); rows=[]
    for k in k_values:
      for h in horizons:
        retcol=f'+{h}D Max Return %'
        if retcol not in x:continue
        top=[]; rest=[]; days=0
        for _,g in x.dropna(subset=[score_col,retcol]).groupby('Date'):
          if len(g)<=k:continue
          g=g.sort_values(score_col,ascending=False); top.extend(g.head(k)[retcol].astype(float)); rest.extend(g.iloc[k:][retcol].astype(float)); days+=1
        if top and rest:
          ta=float(np.mean(top)); ra=float(np.mean(rest)); rows.append({'Score':score_col,'Top K':k,'Horizon':f'{h}D','Days':days,'TopK Avg Max Return %':ta,'Rest Avg Max Return %':ra,'Excess %':ta-ra,'TopK Positive %':100*np.mean(np.array(top)>0)})
    return pd.DataFrame(rows)

def atr_target_validation(feat, threshold=60, atr_multiples=(1.0,1.5,2.0), horizons=(1,3,5)):
    f=feat.copy().dropna(subset=['Close']); rows=[]
    if len(f)<120:return pd.DataFrame()
    scores=np.array([explosive_score_row(r)[0] for _,r in f.iterrows()]); highs=f.High.to_numpy(float); closes=f.Close.to_numpy(float); atrp=f['atr_pct'].to_numpy(float)/100
    for mult in atr_multiples:
      for h in horizons:
        sig=[]; base=[]
        for i in range(int(len(f)*.4),len(f)-h):
          if not np.isfinite(atrp[i]):continue
          hit=float(np.nanmax(highs[i+1:i+1+h])/closes[i]-1 >= mult*atrp[i]); base.append(hit)
          if scores[i]>=threshold:sig.append(hit)
        hr=np.mean(sig) if sig else np.nan; br=np.mean(base) if base else np.nan
        rows.append({'ATR Target':f'+{mult:g} ATR','Horizon':f'{h}D','Signals':len(sig),'Hit Rate %':100*hr if np.isfinite(hr) else np.nan,'Baseline %':100*br if np.isfinite(br) else np.nan,'Lift':hr/br if np.isfinite(hr) and br>0 else np.nan})
    return pd.DataFrame(rows)

def score_row(r, min_turnover=3_000_000):
    comp=[]
    m3=float(r.get('mom3',np.nan)); m5=float(r.get('mom5',np.nan)); s=0
    if np.isfinite(m3) and np.isfinite(m5):
        s += 5 if m3>0 else 0; s += 4 if m5>0 else 0; s += 3 if m3>.02 else 0; s += 2 if m5>.04 else 0
    s=_clamp(s,0,16); comp.append(('Momentum',s,16))
    vr=float(r.get('volume_ratio',np.nan)); vz=float(r.get('volume_z',np.nan)); va=float(r.get('vol_accel',np.nan)); s=0
    if np.isfinite(vr): s+=(5 if vr>=1.2 else 0)+(4 if vr>=1.5 else 0)+(3 if vr>=2 else 0)
    if np.isfinite(vz) and vz>=1:s+=4
    if np.isfinite(va) and va>=1.15:s+=4
    s=_clamp(s,0,20); comp.append(('Volume acceleration',s,20))
    b=float(r.get('breakout20_pct',np.nan)); s=0
    if np.isfinite(b):
        if -3<=b<=0:s=9
        elif 0<b<=4:s=14
        elif 4<b<=8:s=10
    comp.append(('20D breakout/proximity',s,14))
    rv=float(r.get('rsi14',np.nan)); s=8 if np.isfinite(rv) and 52<=rv<=67 else 5 if np.isfinite(rv) and 48<=rv<=72 else 0; comp.append(('RSI',s,8))
    mh=float(r.get('macd_hist',np.nan)); ms=float(r.get('macd_hist_slope',np.nan)); s=(7 if np.isfinite(mh) and mh>0 else 0)+(5 if np.isfinite(ms) and ms>0 else 0); comp.append(('MACD histogram',s,12))
    av=float(r.get('adx14',np.nan)); s=10 if np.isfinite(av) and av>=30 else 8 if np.isfinite(av) and av>=25 else 5 if np.isfinite(av) and av>=20 else 0; comp.append(('ADX trend strength',s,10))
    at=float(r.get('atr_pct',np.nan)); s=8 if np.isfinite(at) and 2<=at<=7 else 5 if np.isfinite(at) and 1.2<=at<=10 else 0; comp.append(('ATR / volatility',s,8))
    cl=float(r.get('close_location',np.nan)); s=6 if np.isfinite(cl) and cl>=.75 else 4 if np.isfinite(cl) and cl>=.55 else 2 if np.isfinite(cl) and cl>=.4 else 0; comp.append(('Candle strength',s,6))
    tv=float(r.get('turnover',np.nan)); tr=float(r.get('turnover_ratio60',np.nan))
    if np.isfinite(tv) and min_turnover > 0 and tv < min_turnover:
        s=0
    elif np.isfinite(tr):
        s=6 if tr>=1.5 else 4 if tr>=1.2 else 2 if tr>=0.8 else 0
    else:
        s=0
    comp.append(('Liquidity',s,6))
    return float(_clamp(sum(p for _,p,_ in comp))), comp


def early_score_row(r):
    pts=[]
    ob=float(r.get('obv_slope5',np.nan)); pts.append(('OBV accumulation',18 if np.isfinite(ob) and ob>.08 else 12 if np.isfinite(ob) and ob>0 else 0,18))
    cm=float(r.get('cmf20',np.nan)); pts.append(('CMF money flow',18 if np.isfinite(cm) and cm>.12 else 12 if np.isfinite(cm) and cm>0 else 0,18))
    ad=float(r.get('ad_slope5',np.nan)); pts.append(('Accumulation/Distribution',12 if np.isfinite(ad) and ad>.05 else 8 if np.isfinite(ad) and ad>0 else 0,12))
    sq=bool(r.get('squeeze',0)); rel=bool(r.get('squeeze_release',0)); pts.append(('Bollinger/Keltner squeeze',14 if rel else 9 if sq else 0,14))
    rs=float(r.get('rs20',np.nan)); pts.append(('Relative strength',12 if np.isfinite(rs) and rs>.08 else 8 if np.isfinite(rs) and rs>0 else 0,12))
    ra=float(r.get('roc_accel',np.nan)); pts.append(('ROC acceleration',10 if np.isfinite(ra) and ra>.02 else 6 if np.isfinite(ra) and ra>0 else 0,10))
    pts.append(('Price/Volume divergence',10 if bool(r.get('pv_divergence',0)) else 0,10))
    vr=float(r.get('volume_ratio',np.nan)); va=float(r.get('vol_accel',np.nan)); pts.append(('Early RVOL',6 if np.isfinite(vr) and .8<=vr<1.5 and np.isfinite(va) and va>1 else 3 if np.isfinite(va) and va>1 else 0,6))
    return float(_clamp(sum(p for _,p,_ in pts))), pts


def score_latest(feat, min_turnover=3_000_000):
    r=feat.dropna(subset=['Close']).iloc[-1]
    score,components=score_row(r,min_turnover)
    early,early_components=early_score_row(r)
    out={'score':score,'early_score':early,'price':float(r['Close']),'daily_change_pct':float(r.get('ret1',np.nan))*100,'components':components,'early_components':early_components}
    for k in ['volume_ratio','rsi14','adx14','atr_pct','cmf20','obv_slope5','ad_slope5','rs20','roc_accel','vwap','ema9','ema20','ema50','support20','resistance20','macd_hist','robust_volume_ratio','post_impulse_retention','volume_dryup','volume_reexpansion','fresh_transition_count']:
        val=r.get(k,np.nan); out[k]=float(val) if not pd.isna(val) else np.nan
    return out


def backtest_signal(feat, horizon=5, target_pct=.06, score_threshold=66, min_turnover=3_000_000):
    f=feat.copy(); f['score']=[score_row(r,min_turnover)[0] for _,r in f.iterrows()]
    highs=f['High'].to_numpy(dtype='float64'); lows=f['Low'].to_numpy(dtype='float64'); closes=f['Close'].to_numpy(dtype='float64')
    hits=np.full(len(f),np.nan); rets=np.full(len(f),np.nan); dds=np.full(len(f),np.nan)
    for i in range(len(f)-horizon):
        fh=highs[i+1:i+1+horizon]; fl=lows[i+1:i+1+horizon]; end=closes[i+horizon]
        hits[i]=1. if np.nanmax(fh)>=closes[i]*(1+target_pct) else 0.
        rets[i]=end/closes[i]-1; dds[i]=np.nanmin(fl)/closes[i]-1
    f['hit']=hits; f['fwd_return']=rets; f['drawdown']=dds
    valid=f[(f['score']>=score_threshold)&f['hit'].notna()].copy(); n=len(valid)
    if n==0:return {'n':0,'hits':0,'misses':0,'hit_rate':np.nan,'avg_return':np.nan,'max_drawdown':np.nan,'confidence':0,'confidence_label':'LOW','backtest_performance':0.0,'sample_reliability':0.0}
    hr=float(valid['hit'].mean())
    sample_factor=min(1.,math.sqrt(n/60))
    # V5.3.1: confidence gives more weight to demonstrated hit-rate performance,
    # while sample size remains an explicit reliability adjustment.
    conf=100*(.80*hr+.20*sample_factor)
    label='HIGH' if n>=30 and hr>=.60 else 'MEDIUM' if n>=12 and hr>=.45 else 'LOW'
    hits=int(valid['hit'].sum())
    return {
        'n':int(n),'hits':hits,'misses':int(n-hits),'hit_rate':hr,
        'avg_return':float(valid['fwd_return'].mean()),'max_drawdown':float(valid['drawdown'].min()),
        'confidence':round(conf,1),'confidence_label':label,
        'backtest_performance':round(hr*100,1),'sample_reliability':round(sample_factor*100,1)
    }


def entry_timing(feat, quant_score, early_score):
    r=feat.dropna(subset=['Close']).iloc[-1]; p=float(r['Close']); atr=float(r.get('atr14',np.nan)); vwap=float(r.get('vwap',np.nan)); e9=float(r.get('ema9',np.nan)); e20=float(r.get('ema20',np.nan)); res=float(r.get('resistance20',np.nan)); sup=float(r.get('support20',np.nan)); vr=float(r.get('volume_ratio',np.nan)); mh=float(r.get('macd_hist',np.nan)); rv=float(r.get('rsi14',np.nan))
    score=0
    if np.isfinite(vwap) and p>=vwap:score+=20
    if np.isfinite(e9) and np.isfinite(e20) and e9>=e20:score+=18
    if np.isfinite(vr) and vr>=1.15:score+=16
    if np.isfinite(mh) and mh>0:score+=14
    if np.isfinite(rv) and 50<=rv<=70:score+=12
    score += min(20,max(0,(quant_score-55)*.8)); score=float(_clamp(score))
    a=atr if np.isfinite(atr) and atr>0 else p*.02
    zone_low=max(x for x in [p-.35*a, vwap if np.isfinite(vwap) else p-.35*a, e9 if np.isfinite(e9) else p-.35*a] if np.isfinite(x))
    zone_high=p+.10*a; trigger=max(p, res if np.isfinite(res) and res<p+1.2*a else p)+.08*a; invalid=max(sup if np.isfinite(sup) else p-1.5*a, p-1.6*a)
    stretched=np.isfinite(e20) and p>e20+1.5*a; near_break=np.isfinite(res) and 0 <= (res-p)/p <= .015
    if stretched: status='WAIT FOR PULLBACK'
    elif near_break and (not np.isfinite(vr) or vr<1.2): status='WAIT FOR BREAKOUT'
    elif score>=72 and quant_score>=66: status='ENTER ZONE'
    else: status='WATCH ENTRY'
    return {'entry_score':round(score,1),'status':status,'zone_low':zone_low,'zone_high':zone_high,'trigger':trigger,'invalidation':invalid,'target1':p*1.03,'target2':p*1.06}


def scan_universe(tickers,daily_period='6mo',use_hourly=True,hourly_period='1mo',horizon=5,target_pct=.06,min_turnover=3_000_000,buy_threshold=66,prefilter_top=30):
    stage1=[]
    for ticker in tickers:
        try:
            d=fetch_ohlcv(ticker,daily_period,'1d')
            if d is None or len(d)<35: continue
            f=compute_features(d); latest=score_latest(f,min_turnover); pre=.72*latest['score']+.28*latest['early_score']
            stage1.append((pre,ticker,d,f,latest))
        except Exception:
            continue
    if not stage1:return pd.DataFrame()
    stage1.sort(key=lambda z:z[0],reverse=True); finalists=stage1[:max(5,min(int(prefilter_top),len(stage1)))]
    rows=[]
    for pre,ticker,d,f,latest in finalists:
        try:
            hs=he=np.nan; hfeat=m15feat=None
            if use_hourly:
                h=fetch_ohlcv(ticker,hourly_period,'1h')
                if h is not None and len(h)>=30:
                    hfeat=compute_features(h,True); hl=score_latest(hfeat,0); hs=hl['score']; he=hl['early_score']
                m15=fetch_ohlcv(ticker,'1mo','15m')
                if m15 is not None and len(m15)>=30:m15feat=compute_features(m15,True)
            final=latest['score'] if not np.isfinite(hs) else .78*latest['score']+.22*hs
            early=latest['early_score'] if not np.isfinite(he) else .70*latest['early_score']+.30*he
            precision_feat=m15feat if m15feat is not None else hfeat if hfeat is not None else f
            ent=entry_timing(precision_feat,final,early); bt=backtest_signal(f,horizon,target_pct,buy_threshold,min_turnover)
            rows.append({'Ticker':ticker,'Score':round(final,1),'EarlyScore':round(early,1),'EntryScore':ent['entry_score'],'EntryStatus':ent['status'],'EntryLow':round(ent['zone_low'],4),'EntryHigh':round(ent['zone_high'],4),'Trigger':round(ent['trigger'],4),'Invalidation':round(ent['invalidation'],4),'Price':round(latest['price'],4),'VolumeRatio':round(latest['volume_ratio'],2) if np.isfinite(latest['volume_ratio']) else np.nan,'RSI14':round(latest['rsi14'],1) if np.isfinite(latest['rsi14']) else np.nan,'ADX14':round(latest['adx14'],1) if np.isfinite(latest['adx14']) else np.nan,'CMF20':round(latest['cmf20'],3) if np.isfinite(latest['cmf20']) else np.nan,'EmpiricalHitRate':round(bt['hit_rate']*100,1) if bt['n'] else np.nan,'BacktestN':bt['n'],'Confidence':bt['confidence_label'],'ConfidenceScore':bt['confidence'],'AvgReturn':round(bt['avg_return']*100,2) if bt['n'] else np.nan,'MaxDrawdown':round(bt['max_drawdown']*100,2) if bt['n'] else np.nan})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values(['Score','EarlyScore','EntryScore'],ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def early_event_backtest(feat, event_pct=.06, lookbacks=(1, 2, 3), min_turnover=0):
    """Analyze Quant/Early scores before single-day close-to-close gains >= event_pct.

    This is intentionally a single-ticker diagnostic. An event is a trading day whose
    Close is at least event_pct above the previous trading day's Close. For each event,
    return the Quant and Early scores 1/2/3 trading days before it.
    """
    f = feat.copy()
    if f is None or len(f) == 0:
        return pd.DataFrame()
    f['quant_score_bt'] = [score_row(r, min_turnover)[0] for _, r in f.iterrows()]
    f['early_score_bt'] = [early_score_row(r)[0] for _, r in f.iterrows()]
    f['day_return_bt'] = f['Close'].pct_change()
    rows = []
    for i in range(1, len(f)):
        day_ret = float(f['day_return_bt'].iloc[i]) if pd.notna(f['day_return_bt'].iloc[i]) else np.nan
        if not np.isfinite(day_ret) or day_ret < float(event_pct):
            continue
        row = {
            'EventDate': f.index[i],
            'EventReturnPct': day_ret * 100.0,
            'PrevClose': float(f['Close'].iloc[i-1]),
            'EventClose': float(f['Close'].iloc[i]),
        }
        for lb in lookbacks:
            j = i - int(lb)
            row[f'Quant_D{lb}'] = float(f['quant_score_bt'].iloc[j]) if j >= 0 else np.nan
            row[f'Early_D{lb}'] = float(f['early_score_bt'].iloc[j]) if j >= 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

# ========================= V5.4 Dynamic Calibration =========================

def _component_dict(r, kind='early', min_turnover=0):
    comps = early_score_row(r)[1] if kind == 'early' else score_row(r, min_turnover)[1]
    return {name: (float(pts), float(mx)) for name, pts, mx in comps}


def calibrate_components(feat, kind='early', event_pct=.03, lookbacks=(1,2,3), min_turnover=0):
    """Ticker-specific, evidence-weighted component calibration.

    Calibration uses historical close-to-close up-events. Multipliers are deliberately
    shrunk toward 1.0 when event samples are small and capped to prevent overfitting.
    """
    f = feat.copy()
    if f is None or len(f) < 35:
        return {'events':0, 'confidence':0.0, 'multipliers':{}, 'table':pd.DataFrame()}
    event_ret = f['Close'].pct_change()
    events = [i for i in range(1, len(f)) if pd.notna(event_ret.iloc[i]) and float(event_ret.iloc[i]) >= float(event_pct)]
    sample_rel = min(1.0, math.sqrt(len(events)/30.0)) if events else 0.0
    baseline = {}
    maxima = {}
    for _, r in f.iterrows():
        for name,(pts,mx) in _component_dict(r,kind,min_turnover).items():
            maxima[name]=mx; baseline.setdefault(name,[]).append(pts>0)
    rows=[]; multipliers={}
    for name,mx in maxima.items():
        pre=[]; seen=set(); by_lb={lb:[] for lb in lookbacks}
        for eno,i in enumerate(events):
            for lb in lookbacks:
                j=i-int(lb)
                if j<0: continue
                active=_component_dict(f.iloc[j],kind,min_turnover).get(name,(0,mx))[0] > 0
                pre.append(active); by_lb[lb].append(active)
                if active: seen.add(eno)
        pre_rate=float(np.mean(pre)) if pre else np.nan
        base_rate=float(np.mean(baseline.get(name,[]))) if baseline.get(name) else np.nan
        lift=pre_rate/base_rate if np.isfinite(pre_rate) and np.isfinite(base_rate) and base_rate>0 else 1.0
        coverage=len(seen)/len(events) if events else 0.0
        lb_rates=[float(np.mean(v)) for v in by_lb.values() if v]
        stability=max(0.55, 1.0-(float(np.std(lb_rates))/0.35)) if lb_rates else 0.55
        # Evidence-shrunk multiplier. High lift + coverage + sample quality earns weight;
        # weak/negative lift loses weight. Hard caps keep one ticker from dominating.
        evidence=sample_rel*(0.55+0.45*coverage)*stability
        mult=1.0 + evidence*(lift-1.0)*1.65
        mult=float(np.clip(mult,0.45,1.80))
        multipliers[name]=mult
        rows.append({'Component':name,'Base Weight':mx,'Dynamic Weight':mx*mult,'Lift x':lift,'Coverage %':coverage*100,'Sample Reliability %':sample_rel*100,'Stability %':stability*100,'Multiplier':mult})
    table=pd.DataFrame(rows)
    if not table.empty: table=table.sort_values(['Dynamic Weight','Lift x'],ascending=False).reset_index(drop=True)
    return {'events':len(events),'confidence':sample_rel*100,'multipliers':multipliers,'table':table}


def dynamic_score_row(r, calibration, kind='early', min_turnover=0):
    comps=_component_dict(r,kind,min_turnover); mults=(calibration or {}).get('multipliers',{})
    num=den=0.0; details=[]
    for name,(pts,mx) in comps.items():
        m=float(mults.get(name,1.0)); dw=mx*m
        strength=(pts/mx) if mx>0 else 0.0
        num += strength*dw; den += dw
        details.append((name,pts,mx,dw,m))
    score=100.0*num/den if den>0 else 0.0
    return float(_clamp(score)), details


def calibrate_combinations(feat, event_pct=.03, lookbacks=(1,2,3), min_turnover=0, top_n=12):
    """Find Early+Quant component pairs whose joint activation has historical lift."""
    f=feat.copy()
    if f is None or len(f)<35:return pd.DataFrame()
    er=f['Close'].pct_change(); events=[i for i in range(1,len(f)) if pd.notna(er.iloc[i]) and float(er.iloc[i])>=float(event_pct)]
    if not events:return pd.DataFrame()
    def active_set(r):
        e={f'E:{n}' for n,(p,m) in _component_dict(r,'early',min_turnover).items() if p>0}
        q={f'Q:{n}' for n,(p,m) in _component_dict(r,'quant',min_turnover).items() if p>0}
        return e|q
    sets=[active_set(r) for _,r in f.iterrows()]
    names=sorted(set().union(*sets)); pairs=[]
    for a_i,a in enumerate(names):
        for b in names[a_i+1:]:
            # only cross-family pairs to keep the search interpretable and reduce overfit
            if a[0]==b[0]: continue
            base=np.mean([(a in s and b in s) for s in sets])
            if base<=0: continue
            pre=[]; seen=set()
            for eno,i in enumerate(events):
                for lb in lookbacks:
                    j=i-int(lb)
                    if j<0:continue
                    on=a in sets[j] and b in sets[j]; pre.append(on)
                    if on:seen.add(eno)
            pr=np.mean(pre) if pre else 0; lift=pr/base if base else np.nan; cov=100*len(seen)/len(events)
            if np.isfinite(lift): pairs.append({'Combination':a[2:]+' + '+b[2:],'Lift x':lift,'Coverage %':cov,'Pre-event active %':100*pr,'Baseline active %':100*base})
    z=pd.DataFrame(pairs)
    if z.empty:return z
    return z.sort_values(['Lift x','Coverage %'],ascending=False).head(top_n).reset_index(drop=True)


def dynamic_scores(feat, event_pct=.03, min_turnover=0):
    """Current static vs dynamically calibrated scores for one ticker."""
    r=feat.dropna(subset=['Close']).iloc[-1]
    ec=calibrate_components(feat,'early',event_pct,(1,2,3),min_turnover)
    qc=calibrate_components(feat,'quant',event_pct,(1,2,3),min_turnover)
    se=early_score_row(r)[0]; sq=score_row(r,min_turnover)[0]
    de,ed=dynamic_score_row(r,ec,'early',min_turnover); dq,qd=dynamic_score_row(r,qc,'quant',min_turnover)
    # Dynamic final weighting: base 35% Early, shifted modestly by relative calibration evidence.
    er=float(ec.get('confidence',0))/100; qr=float(qc.get('confidence',0))/100
    early_w=float(np.clip(.35 + .10*(er-qr),.25,.45)); quant_w=1.0-early_w
    pred=quant_w*dq+early_w*de
    return {'static_early':se,'static_quant':sq,'dynamic_early':de,'dynamic_quant':dq,'final_prediction':float(_clamp(pred)),'early_weight':early_w,'quant_weight':quant_w,'early_calibration':ec,'quant_calibration':qc,'combinations':calibrate_combinations(feat,event_pct,(1,2,3),min_turnover)}


def _forward_outcomes(f,horizon,target_pct):
    highs=f['High'].to_numpy(dtype='float64'); lows=f['Low'].to_numpy(dtype='float64'); closes=f['Close'].to_numpy(dtype='float64')
    hit=np.full(len(f),np.nan); ret=np.full(len(f),np.nan); dd=np.full(len(f),np.nan)
    for i in range(len(f)-horizon):
        hit[i]=1. if np.nanmax(highs[i+1:i+1+horizon])>=closes[i]*(1+target_pct) else 0.
        ret[i]=closes[i+horizon]/closes[i]-1; dd[i]=np.nanmin(lows[i+1:i+1+horizon])/closes[i]-1
    return hit,ret,dd


def compare_static_dynamic_backtest(feat,horizon=5,target_pct=.03,score_threshold=66,min_turnover=0,train_fraction=.70):
    """Leakage-reduced holdout comparison: calibrate on earlier history, evaluate later history."""
    f=feat.copy().dropna(subset=['Close'])
    if len(f)<60:return {'static':{},'dynamic':{},'baseline':np.nan,'signal_lift_static':np.nan,'signal_lift_dynamic':np.nan,'train_rows':0,'test_rows':0}
    cut=max(35,min(len(f)-int(horizon)-10,int(len(f)*train_fraction))); train=f.iloc[:cut].copy(); test=f.iloc[cut:].copy()
    ec=calibrate_components(train,'early',target_pct,(1,2,3),min_turnover); qc=calibrate_components(train,'quant',target_pct,(1,2,3),min_turnover)
    hit,ret,dd=_forward_outcomes(f,int(horizon),float(target_pct)); f['_hit']=hit; f['_ret']=ret; f['_dd']=dd
    rows=[]
    for i in range(cut,len(f)):
        if not np.isfinite(hit[i]):continue
        r=f.iloc[i]; ss=score_row(r,min_turnover)[0]; de,_=dynamic_score_row(r,ec,'early',min_turnover); dq,_=dynamic_score_row(r,qc,'quant',min_turnover)
        ew=float(np.clip(.35+.10*((ec.get('confidence',0)-qc.get('confidence',0))/100),.25,.45)); pred=(1-ew)*dq+ew*de
        rows.append((ss,pred,hit[i],ret[i],dd[i]))
    if not rows:return {'static':{},'dynamic':{},'baseline':np.nan,'signal_lift_static':np.nan,'signal_lift_dynamic':np.nan,'train_rows':len(train),'test_rows':0}
    z=pd.DataFrame(rows,columns=['static','dynamic','hit','ret','dd']); baseline=float(z.hit.mean())
    def pack(col):
        v=z[z[col]>=score_threshold]; n=len(v)
        if not n:return {'n':0,'hits':0,'hit_rate':np.nan,'avg_return':np.nan,'max_drawdown':np.nan,'sample_reliability':0.0}
        return {'n':n,'hits':int(v.hit.sum()),'hit_rate':float(v.hit.mean()),'avg_return':float(v.ret.mean()),'max_drawdown':float(v.dd.min()),'sample_reliability':100*min(1.,math.sqrt(n/60))}
    s=pack('static'); d=pack('dynamic')
    return {'static':s,'dynamic':d,'baseline':baseline,'signal_lift_static':(s.get('hit_rate',np.nan)/baseline if baseline>0 and np.isfinite(s.get('hit_rate',np.nan)) else np.nan),'signal_lift_dynamic':(d.get('hit_rate',np.nan)/baseline if baseline>0 and np.isfinite(d.get('hit_rate',np.nan)) else np.nan),'train_rows':len(train),'test_rows':len(z)}

# V5.4.3 dynamic scanner follows.

def scan_universe_dynamic(tickers,daily_period='6mo',use_hourly=True,hourly_period='1mo',horizon=5,target_pct=.03,min_turnover=0,buy_threshold=66,prefilter_top=30):
    """V5.4 scanner: dynamic ticker calibration + prediction rank + entry timing."""
    stage=[]
    for ticker in tickers:
        try:
            d=fetch_ohlcv(ticker,daily_period,'1d')
            if d is None or len(d)<35:continue
            f=compute_features(d); ds=dynamic_scores(f,target_pct,min_turnover)
            # Fast ranking favors calibrated prediction, with a small static anchor.
            pre=.85*ds['final_prediction']+.15*(.72*ds['static_quant']+.28*ds['static_early'])
            stage.append((pre,ticker,f,ds))
        except Exception:continue
    if not stage:return pd.DataFrame()
    stage.sort(key=lambda x:x[0],reverse=True); stage=stage[:max(5,min(int(prefilter_top),len(stage)))]
    rows=[]
    for pre,ticker,f,ds in stage:
        try:
            # Intraday remains a timing/confirmation layer; daily calibration remains the evidence base.
            hfeat=m15feat=None; hq=he=np.nan
            if use_hourly:
                h=fetch_ohlcv(ticker,hourly_period,'1h')
                if h is not None and len(h)>=30:
                    hfeat=compute_features(h,True); hr=hfeat.dropna(subset=['Close']).iloc[-1]
                    hq,_=dynamic_score_row(hr,ds['quant_calibration'],'quant',0); he,_=dynamic_score_row(hr,ds['early_calibration'],'early',0)
                m15=fetch_ohlcv(ticker,'1mo','15m')
                if m15 is not None and len(m15)>=30:m15feat=compute_features(m15,True)
            dq=ds['dynamic_quant'] if not np.isfinite(hq) else .78*ds['dynamic_quant']+.22*hq
            de=ds['dynamic_early'] if not np.isfinite(he) else .70*ds['dynamic_early']+.30*he
            ew=ds['early_weight']; pred=(1-ew)*dq+ew*de
            precision=m15feat if m15feat is not None else hfeat if hfeat is not None else f
            ent=entry_timing(precision,dq,de)
            cmp=compare_static_dynamic_backtest(f,horizon,target_pct,buy_threshold,min_turnover)
            db=cmp.get('dynamic',{}); conf=float(ds['early_calibration'].get('confidence',0)+ds['quant_calibration'].get('confidence',0))/2
            lr=f.dropna(subset=['Close']).iloc[-1]
            rows.append({'Ticker':ticker,'Prediction':round(pred,1),'DynamicQuant':round(dq,1),'DynamicEarly':round(de,1),'StaticQuant':round(ds['static_quant'],1),'StaticEarly':round(ds['static_early'],1),'EntryScore':ent['entry_score'],'EntryStatus':ent['status'],'SignalLift':round(cmp.get('signal_lift_dynamic',np.nan),2) if np.isfinite(cmp.get('signal_lift_dynamic',np.nan)) else np.nan,'CalibrationConfidence':round(conf,1),'BacktestN':int(db.get('n',0) or 0),'EmpiricalHitRate':round(float(db.get('hit_rate'))*100,1) if db.get('n',0) and np.isfinite(db.get('hit_rate',np.nan)) else np.nan,'Price':round(float(lr['Close']),4),'VolumeRatio':round(float(lr.get('volume_ratio',np.nan)),2) if np.isfinite(float(lr.get('volume_ratio',np.nan))) else np.nan,'RSI14':round(float(lr.get('rsi14',np.nan)),1) if np.isfinite(float(lr.get('rsi14',np.nan))) else np.nan})
        except Exception:continue
    return pd.DataFrame(rows).sort_values(['Prediction','DynamicEarly','EntryScore'],ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
