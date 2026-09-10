
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_WEIGHTS = {
    "momentum": 16,
    "volume": 20,
    "breakout": 14,
    "rsi": 8,
    "macd": 12,
    "adx": 10,
    "atr": 8,
    "candle": 6,
    "liquidity": 6,
}

def fetch_ohlcv(ticker: str, period="6mo", interval="1d"):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance can return MultiIndex even for a single ticker
            if ticker in df.columns.get_level_values(-1):
                try:
                    df = df.xs(ticker, axis=1, level=-1)
                except Exception:
                    pass
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        needed = ["Open","High","Low","Close","Volume"]
        if not all(c in df.columns for c in needed):
            return None
        out = df[needed].copy()
        clean = pd.DataFrame(index=out.index)
        for col in needed:
            v = out[col]
            if isinstance(v, pd.DataFrame):
                if v.shape[1] == 0:
                    return None
                v = v.iloc[:, 0]
            clean[col] = pd.to_numeric(v, errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        return clean
    except Exception:
        return None

def _as_numeric_series(obj, index=None):
    """Return a clean 1-D numeric Series even if upstream data is a 1-col DataFrame."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(dtype=float, index=index)
        obj = obj.iloc[:, 0]
    if not isinstance(obj, pd.Series):
        obj = pd.Series(obj, index=index)
    return pd.to_numeric(obj, errors="coerce").astype(float)

def _safe_float(v, default=np.nan):
    """Convert scalars/1-item Series safely; never raise TypeError on pandas objects."""
    try:
        if isinstance(v, pd.DataFrame):
            if v.empty:
                return float(default)
            v = v.iloc[-1, 0]
        elif isinstance(v, pd.Series):
            if v.empty:
                return float(default)
            v = v.iloc[-1]
        a = np.asarray(v)
        if a.size == 0:
            return float(default)
        return float(a.reshape(-1)[-1])
    except (TypeError, ValueError, IndexError):
        return float(default)

def ema(s, span):
    s = _as_numeric_series(s)
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(close, n=14):
    close = _as_numeric_series(close)
    d = close.diff()
    gain = d.clip(lower=0.0)
    loss = (-d.clip(upper=0.0))
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    # Avoid pandas .replace() dtype edge-cases seen on Streamlit Cloud.
    denom = al.where(al.ne(0.0))
    rs = ag.div(denom)
    return 100.0 - (100.0 / (1.0 + rs))

def true_range(df):
    prev = df["Close"].shift(1)
    return pd.concat([
        df["High"]-df["Low"],
        (df["High"]-prev).abs(),
        (df["Low"]-prev).abs()
    ], axis=1).max(axis=1)

def adx(df, n=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df)
    atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (plus_di-minus_di).abs() / (plus_di + minus_di).where((plus_di + minus_di).ne(0.0))
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def compute_features(df: pd.DataFrame, intraday=False):
    x = df.copy()
    # Defensive normalization: every OHLCV field must be a 1-D numeric Series.
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in x.columns:
            raise ValueError(f"Missing required column: {col}")
        x[col] = _as_numeric_series(x[col], index=x.index)
    x = x.replace([np.inf, -np.inf], np.nan)
    c = x["Close"]
    x["ret1"] = c.pct_change()
    x["mom3"] = c.pct_change(3)
    x["mom5"] = c.pct_change(5)
    x["mom10"] = c.pct_change(10)

    vol_mean20 = x["Volume"].rolling(20).mean()
    vol_std20 = x["Volume"].rolling(20).std()
    x["volume_ratio"] = x["Volume"] / vol_mean20
    x["volume_z"] = (x["Volume"] - vol_mean20) / vol_std20.where(vol_std20.ne(0.0))
    x["vol_accel"] = x["Volume"].rolling(3).mean() / x["Volume"].rolling(10).mean()

    x["rsi14"] = rsi(c, 14)
    macd = ema(c, 12) - ema(c, 26)
    signal = ema(macd, 9)
    x["macd"] = macd
    x["macd_hist"] = macd - signal
    x["macd_hist_slope"] = x["macd_hist"].diff(2)

    tr = true_range(x)
    x["atr14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    x["atr_pct"] = x["atr14"] / c * 100
    x["adx14"] = adx(x, 14)

    prior_high20 = x["High"].rolling(20).max().shift(1)
    x["breakout20_pct"] = (c / prior_high20 - 1) * 100
    x["dist_to_high20_pct"] = (prior_high20 / c - 1) * 100

    rng = (x["High"] - x["Low"]).where((x["High"] - x["Low"]).ne(0.0))
    x["close_location"] = (c - x["Low"]) / rng
    x["turnover"] = c * x["Volume"]

    # "freshness": reward recent acceleration rather than old trend only
    x["fresh_momentum"] = x["mom3"] - x["mom10"] / 3.0
    return x

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def score_row(r, min_turnover=3_000_000):
    components = []

    # Momentum 0..16: positive but avoid extremely stretched moves.
    m3, m5 = _safe_float(r.get("mom3", np.nan)), _safe_float(r.get("mom5", np.nan))
    momentum = 0
    if np.isfinite(m3) and np.isfinite(m5):
        if m3 > 0: momentum += 5
        if m5 > 0: momentum += 4
        if m3 > 0.02: momentum += 3
        if m5 > 0.04: momentum += 2
        if 0 < m3 < 0.12: momentum += 2
        if m3 > 0.20: momentum -= 4
    momentum = _clamp(momentum, 0, 16)
    components.append(("Momentum", momentum, 16))

    vr = _safe_float(r.get("volume_ratio", np.nan))
    vz = _safe_float(r.get("volume_z", np.nan))
    va = _safe_float(r.get("vol_accel", np.nan))
    volume = 0
    if np.isfinite(vr):
        if vr >= 1.2: volume += 5
        if vr >= 1.5: volume += 4
        if vr >= 2.0: volume += 3
    if np.isfinite(vz) and vz >= 1: volume += 4
    if np.isfinite(va) and va >= 1.15: volume += 4
    volume = _clamp(volume, 0, 20)
    components.append(("Volume acceleration", volume, 20))

    b = _safe_float(r.get("breakout20_pct", np.nan))
    breakout = 0
    if np.isfinite(b):
        if -3 <= b <= 0: breakout = 9
        elif 0 < b <= 4: breakout = 14
        elif 4 < b <= 8: breakout = 10
        elif -6 <= b < -3: breakout = 5
    components.append(("20D breakout/proximity", breakout, 14))

    rv = _safe_float(r.get("rsi14", np.nan))
    rs = 0
    if np.isfinite(rv):
        if 52 <= rv <= 67: rs = 8
        elif 48 <= rv < 52 or 67 < rv <= 72: rs = 5
        elif 40 <= rv < 48: rs = 2
        elif rv > 80: rs = 0
    components.append(("RSI", rs, 8))

    mh = _safe_float(r.get("macd_hist", np.nan))
    ms = _safe_float(r.get("macd_hist_slope", np.nan))
    macd = 0
    if np.isfinite(mh) and mh > 0: macd += 7
    if np.isfinite(ms) and ms > 0: macd += 5
    components.append(("MACD histogram", macd, 12))

    av = _safe_float(r.get("adx14", np.nan))
    ad = 0
    if np.isfinite(av):
        if av >= 30: ad = 10
        elif av >= 25: ad = 8
        elif av >= 20: ad = 5
    components.append(("ADX trend strength", ad, 10))

    at = _safe_float(r.get("atr_pct", np.nan))
    atrs = 0
    if np.isfinite(at):
        if 2.0 <= at <= 7.0: atrs = 8
        elif 1.2 <= at < 2.0 or 7.0 < at <= 10.0: atrs = 5
        elif at > 12: atrs = 1
    components.append(("ATR / volatility", atrs, 8))

    cl = _safe_float(r.get("close_location", np.nan))
    candle = 0
    if np.isfinite(cl):
        if cl >= 0.75: candle = 6
        elif cl >= 0.55: candle = 4
        elif cl >= 0.40: candle = 2
    components.append(("Candle strength", candle, 6))

    tv = _safe_float(r.get("turnover", np.nan))
    liq = 0
    if np.isfinite(tv):
        if tv >= min_turnover * 3: liq = 6
        elif tv >= min_turnover: liq = 4
        elif tv >= min_turnover * .5: liq = 2
    components.append(("Liquidity", liq, 6))

    score = sum(p for _,p,_ in components)
    return float(_clamp(score, 0, 100)), components

def score_latest(feat: pd.DataFrame, min_turnover=3_000_000):
    r = feat.dropna(subset=["Close"]).iloc[-1]
    score, components = score_row(r, min_turnover=min_turnover)
    return {
        "score": score,
        "price": _safe_float(r["Close"]),
        "volume_ratio": _safe_float(r.get("volume_ratio", np.nan)),
        "volume_z": _safe_float(r.get("volume_z", np.nan)),
        "rsi14": _safe_float(r.get("rsi14", np.nan)),
        "adx14": _safe_float(r.get("adx14", np.nan)),
        "atr_pct": _safe_float(r.get("atr_pct", np.nan)),
        "breakout20_pct": _safe_float(r.get("breakout20_pct", np.nan)),
        "daily_change_pct": _safe_float(r.get("ret1", np.nan))*100,
        "turnover": _safe_float(r.get("turnover", np.nan)),
        "components": components,
    }

def backtest_signal(feat: pd.DataFrame, horizon=5, target_pct=0.06,
                    score_threshold=67, min_turnover=3_000_000):
    """
    Walk-forward-like evaluation:
    score at each date uses only rolling features available by that date.
    A hit occurs if a future HIGH within 'horizon' bars reaches target_pct.
    """
    f = feat.copy()
    scores = []
    for _, row in f.iterrows():
        sc, _ = score_row(row, min_turnover=min_turnover)
        scores.append(sc)
    f["score"] = scores

    highs = f["High"].to_numpy()
    closes = f["Close"].to_numpy()
    hit = np.full(len(f), np.nan)
    for i in range(len(f)-horizon):
        future_max = np.nanmax(highs[i+1:i+1+horizon])
        hit[i] = 1.0 if future_max >= closes[i]*(1+target_pct) else 0.0
    f["hit"] = hit

    valid = f[(f["score"] >= score_threshold) & f["hit"].notna()].copy()
    if valid.empty:
        return {"n": 0, "hit_rate": np.nan}
    return {"n": int(len(valid)), "hit_rate": float(valid["hit"].mean())}

def scan_universe(tickers, daily_period="6mo", use_hourly=True, hourly_period="1mo",
                  horizon=5, target_pct=0.06, min_turnover=3_000_000):
    rows = []
    for ticker in tickers:
        d = fetch_ohlcv(ticker, period=daily_period, interval="1d")
        if d is None or len(d) < 35:
            continue
        f = compute_features(d)
        latest = score_latest(f, min_turnover=min_turnover)

        hscore = np.nan
        if use_hourly:
            h = fetch_ohlcv(ticker, period=hourly_period, interval="1h")
            if h is not None and len(h) >= 30:
                hf = compute_features(h, intraday=True)
                hscore = score_latest(hf, min_turnover=0)["score"]

        # Blend daily with hourly confirmation, but do not let hourly dominate.
        final_score = latest["score"]
        if np.isfinite(hscore):
            final_score = 0.78*final_score + 0.22*hscore

        bt = backtest_signal(
            f, horizon=horizon, target_pct=target_pct,
            score_threshold=max(55, final_score-8),
            min_turnover=min_turnover,
        )

        rows.append({
            "Ticker": ticker,
            "Score": round(final_score,1),
            "Price": round(latest["price"],4),
            "DailyChangePct": round(latest["daily_change_pct"],2),
            "VolumeRatio": round(latest["volume_ratio"],2) if np.isfinite(latest["volume_ratio"]) else np.nan,
            "VolumeZ": round(latest["volume_z"],2) if np.isfinite(latest["volume_z"]) else np.nan,
            "RSI14": round(latest["rsi14"],1) if np.isfinite(latest["rsi14"]) else np.nan,
            "ADX14": round(latest["adx14"],1) if np.isfinite(latest["adx14"]) else np.nan,
            "ATRpct": round(latest["atr_pct"],2) if np.isfinite(latest["atr_pct"]) else np.nan,
            "Breakout20Pct": round(latest["breakout20_pct"],2) if np.isfinite(latest["breakout20_pct"]) else np.nan,
            "HourlyScore": round(hscore,1) if np.isfinite(hscore) else np.nan,
            "EmpiricalHitRate": round(bt["hit_rate"]*100,1) if bt["n"] else np.nan,
            "BacktestN": bt["n"],
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
