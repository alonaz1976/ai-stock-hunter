from dataclasses import dataclass
import numpy as np
import pandas as pd


def _colmap(df):
    m = {str(c).lower(): c for c in df.columns}
    req = ['open','high','low','close','volume']
    missing = [x for x in req if x not in m]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    return m


def ema(s,n): return s.ewm(span=n, adjust=False).mean()

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    avg_up=up.ewm(alpha=1/n, adjust=False).mean()
    avg_dn=dn.ewm(alpha=1/n, adjust=False).mean().replace(0,np.nan)
    rs=avg_up/avg_dn
    return 100-(100/(1+rs))

def atr(h,l,c,n=14):
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def cmf(h,l,c,v,n=20):
    rng=(h-l).replace(0,np.nan)
    mfv=(((c-l)-(h-c))/rng)*v
    return mfv.rolling(n).sum()/v.rolling(n).sum()

def obv(c,v):
    return (np.sign(c.diff()).fillna(0)*v).cumsum()

def clamp(x,a=0,b=100):
    if pd.isna(x): return float(a)
    return float(max(a,min(b,x)))

@dataclass
class ScanResult:
    ticker:str; smart_money:float; entry_score:float; stock_score:float
    rsi:float; rvol:float; cmf:float; early_accumulation:bool
    entry_low:float; entry_high:float; invalidation:float; target1:float; target2:float
    verdict:str; price:float; ema20:float; ema50:float; ema200:float; macd_bullish:bool


def analyze(df:pd.DataFrame,ticker='TICKER'):
    df=df.copy().dropna(how='all')
    m=_colmap(df)
    o,h,l,c,v=[pd.to_numeric(df[m[x]], errors='coerce') for x in ['open','high','low','close','volume']]
    valid = c.notna() & h.notna() & l.notna() & v.notna()
    o,h,l,c,v = [s.loc[valid].reset_index(drop=True) for s in [o,h,l,c,v]]
    if len(c)<210: raise ValueError('Need at least 210 daily bars')

    e20,e50,e200=ema(c,20),ema(c,50),ema(c,200)
    macd=ema(c,12)-ema(c,26); sig=ema(macd,9); hist=macd-sig
    R=rsi(c); A=atr(h,l,c); C=cmf(h,l,c,v); O=obv(c,v)
    RV=v/v.rolling(20).mean()
    i=-1
    price=float(c.iloc[i])
    av=A.iloc[i]
    if pd.isna(av) or av<=0:
        av=float((h-l).tail(14).mean())
    if pd.isna(av) or av<=0:
        av=max(price*0.02,0.01)

    sm=0
    sm += clamp((RV.iloc[i]-0.8)/1.7*30,0,30)
    sm += clamp((C.iloc[i]+0.10)/0.35*25,0,25)
    base=abs(float(O.iloc[-21]))+1e-9
    obv_slope=(float(O.iloc[i])-float(O.iloc[-21]))/base
    sm += clamp((obv_slope+0.02)/0.12*25,0,25)
    upvol=v[c.diff()>0].tail(10).mean(); dnvol=v[c.diff()<0].tail(10).mean()
    ratio=(upvol/(dnvol+1e-9)) if pd.notna(upvol) and pd.notna(dnvol) else 1
    sm += clamp((ratio-0.7)/1.3*20,0,20)
    sm=clamp(sm)

    ent=0
    ent += 15 if price>float(e200.iloc[i]) else 0
    ent += 12 if float(e20.iloc[i])>float(e50.iloc[i]) else 0
    ent += 8 if float(e50.iloc[i])>float(e200.iloc[i]) else 0
    macd_bull=bool(macd.iloc[i]>sig.iloc[i])
    ent += 15 if macd_bull and hist.iloc[i]>hist.iloc[-2] else 5 if macd_bull else 0
    rr=float(R.iloc[i]) if pd.notna(R.iloc[i]) else 50.0
    ent += 18 if 48<=rr<=65 else 10 if 40<=rr<70 else 0
    dist=(price/float(e20.iloc[i])-1)
    ent += 17 if -0.02<=dist<=0.04 else 8 if -0.05<=dist<=0.08 else 0
    ent += sm*0.15
    if rr>75: ent-=15
    if dist>0.12: ent-=18
    ent=clamp(ent)

    early=bool(sm>=70 and pd.notna(C.iloc[i]) and C.iloc[i]>0.05 and obv_slope>0 and abs(dist)<=0.06 and rr<70)
    stock=clamp(0.55*sm+0.45*ent)

    support=min(float(e20.iloc[i]), float(c.tail(20).min())+0.5*av)
    entry_low=max(support, price-0.6*av)
    entry_high=price+0.15*av
    invalid=entry_low-1.25*av
    mid=(entry_low+entry_high)/2
    risk=max(mid-invalid, 1e-9)
    t1=mid+2*risk; t2=mid+3*risk
    verdict='BUY ZONE' if ent>=85 and sm>=70 else 'WATCH / WAIT' if ent>=65 else 'NO ENTRY'

    cmf_val=float(C.iloc[i]) if pd.notna(C.iloc[i]) else 0.0
    rvol_val=float(RV.iloc[i]) if pd.notna(RV.iloc[i]) else 0.0
    return ScanResult(
        ticker,round(sm,1),round(ent,1),round(stock,1),round(rr,1),round(rvol_val,2),round(cmf_val,3),early,
        round(entry_low,2),round(entry_high,2),round(invalid,2),round(t1,2),round(t2,2),verdict,
        round(price,2),round(float(e20.iloc[i]),2),round(float(e50.iloc[i]),2),round(float(e200.iloc[i]),2),macd_bull
    )


def backtest(df,ticker='TICKER',min_smart=70,min_entry=80,horizons=(1,5,10,20)):
    rows=[]
    for end in range(210,len(df)-max(horizons)):
        try: r=analyze(df.iloc[:end+1],ticker)
        except Exception: continue
        if r.smart_money>=min_smart and r.entry_score>=min_entry:
            ccol=_colmap(df)['close']; p=float(df.iloc[end][ccol])
            rec={'index':end,'price':p,'smart_money':r.smart_money,'entry_score':r.entry_score}
            for h in horizons: rec[f'ret_{h}d']=float(df.iloc[end+h][ccol]/p-1)
            rows.append(rec)
    out=pd.DataFrame(rows)
    if out.empty: return out, {'signals':0}
    stats={'signals':len(out)}
    for h in horizons:
        x=out[f'ret_{h}d']; stats[f'win_{h}d']=round(float((x>0).mean()),3); stats[f'avg_{h}d']=round(float(x.mean()),4)
    return out,stats
