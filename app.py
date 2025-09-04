import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
import streamlit as st
import math
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import Ridge

st.set_page_config(page_title="Midcap-100 Screener", layout="wide")
st.title("🧮 Nifty Midcap-100 Screener — Fast/Full Modes")

# ---------------- Utils ----------------
def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@st.cache_data(ttl=86400)
def fetch_midcap100_tickers():
    """Fetch Midcap 100 list quickly; fallback to full static list if request hangs/fails."""
    url = "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        tickers = df["Symbol"].astype(str).str.strip().apply(lambda s: f"{s}.NS").tolist()
        names = df["Company Name"].astype(str).str.strip().tolist()
        return list(zip(names, tickers))
    except Exception:
        fallback_symbols = [
            "ABBOTINDIA","ALKEM","ASHOKLEY","AUBANK","AUROPHARMA","BALKRISIND","BEL","BERGEPAINT","BHEL","CANFINHOME",
            "CUMMINSIND","DALBHARAT","DEEPAKNTR","FEDERALBNK","GODREJPROP","HAVELLS","HINDZINC","IDFCFIRSTB","INDHOTEL",
            "INDIAMART","IPCALAB","JUBLFOOD","LUPIN","MANKIND","MUTHOOTFIN","NMDC","OBEROIRLTY","PAGEIND","PERSISTENT",
            "PFIZER","POLYCAB","RECLTD","SAIL","SRF","SUNTV","TATAELXSI","TATAPOWER","TRENT","TVSMOTOR","UBL","VOLTAS",
            "ZYDUSLIFE","PIIND","CONCOR","APOLLOTYRE","TORNTPHARM","MPHASIS","ASTRAL","OFSS","MINDTREE","CROMPTON",
            "ATGL","PETRONET","LTTS","ESCORTS","INDIGO","COLPAL","GILLETTE","BANKBARODA","EXIDEIND","IDBI","INDUSINDBK",
            "LICHSGFIN","MRF","NAVINFLUOR","PFC","PNB","RAMCOCEM","RBLBANK","SHREECEM","TATACHEM","TATACOMM","TORNTPOWER",
            "UNIONBANK","WHIRLPOOL","ZEEL","ABB","ADANIPOWER","AMBUJACEM","BANDHANBNK","CANBK","CHOLAFIN","DLF","EICHERMOT",
            "HINDPETRO","IOC","JINDALSTEL","JSWENERGY","LICI","NTPC","ONGC","POWERGRID","SBICARD","SBILIFE","SBIN","TATAMOTORS",
            "TATASTEEL","TECHM","UPL","VEDL","WIPRO"
        ]
        return [(s, f"{s}.NS") for s in fallback_symbols]

@st.cache_data(show_spinner=False)
def fetch_history(tkr: str, years: int = 5) -> pd.DataFrame:
    """Cached daily history (Close/Adj Close -> y)."""
    df = yf.download(tkr, period=f"{years}y", auto_adjust=False, progress=False, threads=True)
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if not price_col: return pd.DataFrame()
    df = df[["Date", price_col]].rename(columns={"Date":"ds", price_col:"y"}).dropna()
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Light technical features for ML."""
    d = df.copy()
    d["ret_1d"]   = d["y"].pct_change(1)
    d["ret_5d"]   = d["y"].pct_change(5)
    d["ret_21d"]  = d["y"].pct_change(21)
    d["ret_63d"]  = d["y"].pct_change(63)
    d["ret_252d"] = d["y"].pct_change(252)
    d["vol_21d"]  = d["ret_1d"].rolling(21).std()
    delta = d["y"].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean().replace(0, np.nan))
    d["rsi_14"] = 100 - (100 / (1 + rs))
    d["target_30d"]  = d["y"].shift(-30) / d["y"] - 1.0
    d["target_365d"] = d["y"].shift(-365) / d["y"] - 1.0
    return d.dropna().reset_index(drop=True)

def ml_returns(df: pd.DataFrame):
    """Ridge regression predictions for 30d/365d returns (%) from features."""
    if len(df) < 400: return None, None
    d = make_features(df)
    feats = ["ret_5d","ret_21d","ret_63d","ret_252d","vol_21d","rsi_14"]
    if d.empty: return None, None
    X, y30, y365 = d[feats], d["target_30d"], d["target_365d"]
    x_curr = X.iloc[[-1]]
    try:
        m30 = Ridge(alpha=1.0).fit(X.iloc[:-1], y30.iloc[:-1])
        m365 = Ridge(alpha=1.0).fit(X.iloc[:-1], y365.iloc[:-1])
        return float(m30.predict(x_curr)[0])*100.0, float(m365.predict(x_curr)[0])*100.0
    except Exception:
        return None, None

def prophet_ret_prices(df_daily: pd.DataFrame):
    """Weekly downsampled Prophet for speed -> returns & predicted prices."""
    # Weekly downsample (last price of week)
    dfw = df_daily.set_index("ds").resample("W").last().reset_index()
    if len(dfw) < 80: return None  # not enough for a stable weekly model
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
    m.fit(dfw.rename(columns={"ds":"ds","y":"y"}))
    future = m.make_future_dataframe(periods=53, freq="W")  # ~1 year ahead weekly
    fcw = m.predict(future)
    # Map +4w (~1M) and +52w (~1Y) to daily horizon
    curr = float(dfw["y"].iloc[-1])
    row_4w  = fcw.iloc[-4]
    row_52w = fcw.iloc[-1]
    pred1m  = float(row_4w["yhat"])
    pred1y  = float(row_52w["yhat"])
    ret1m   = (pred1m - curr)/curr*100.0
    ret1y   = (pred1y - curr)/curr*100.0
    # Probabilities from weekly intervals
    def p_up(row): 
        yhat, lo, hi = row["yhat"], row.get("yhat_lower", np.nan), row.get("yhat_upper", np.nan)
        if np.isnan(lo) or np.isnan(hi) or hi<=lo: return 1.0 if yhat>curr else (0.5 if abs(yhat-curr)<1e-9 else 0.0)
        sigma = (hi-lo)/(2*1.96);  z = (yhat-curr)/sigma if sigma>0 else 0
        return 1.0 - normal_cdf(-z)
    p1m = p_up(row_4w); p1y = p_up(row_52w)
    return {
        "curr": curr, "prop_ret_1m": ret1m, "prop_ret_1y": ret1y,
        "prop_pred_1m": pred1m, "prop_pred_1y": pred1y,
        "p_up_1m": p1m, "p_up_1y": p1y
    }

def process_one(name_tkr, mode, risk_adj):
    """Compute row for a single ticker (Fast = ML only; Full = blend Prophet+ML)."""
    name, tkr = name_tkr
    hist = fetch_history(tkr)
    if hist.empty or len(hist) < 260:
        return None
    current = float(hist["y"].iloc[-1])

    # ML
    ml1m, ml1y = ml_returns(hist)

    # Defaults
    blended_ret_1m = blended_ret_1y = None
    pred_price_1m = pred_price_1y = None
    p_up_1m = p_up_1y = 0.5

    if mode == "Fast":
        # Use ML only; if ML missing, simple momentum fallback
        if ml1m is None or ml1y is None:
            # 21d momentum fallback
            mom21 = (hist["y"].iloc[-1] / hist["y"].iloc[-21] - 1.0) * 100.0 if len(hist) > 30 else 0.0
            ml1m = mom21 * 0.8
            ml1y = mom21 * 3.0
        blended_ret_1m, blended_ret_1y = ml1m, ml1y
        # crude prob= share of up days last 60 sessions
        up_ratio = (hist["y"].pct_change().tail(60) > 0).mean()
        p_up_1m = float(up_ratio)
        p_up_1y = float(max(min(up_ratio + 0.05, 0.95), 0.05))
    else:
        # Full: Prophet weekly + ML blend 50/50
        p = prophet_ret_prices(hist)
        if p is None:
            # fallback to ML if Prophet failed
            if ml1m is None or ml1y is None: return None
            blended_ret_1m, blended_ret_1y = ml1m, ml1y
            p_up_1m = p_up_1y = 0.6
        else:
            prop_r1m, prop_r1y = p["prop_ret_1m"], p["prop_ret_1y"]
            if ml1m is None or ml1y is None:
                blended_ret_1m, blended_ret_1y = prop_r1m, prop_r1y
            else:
                blended_ret_1m = 0.5*prop_r1m + 0.5*ml1m
                blended_ret_1y = 0.5*prop_r1y + 0.5*ml1y
            p_up_1m, p_up_1y = p["p_up_1m"], p["p_up_1y"]

    # Risk adjustment and price conversion
    blended_ret_1m *= (1 + risk_adj/100.0)
    blended_ret_1y *= (1 + risk_adj/100.0)
    pred_price_1m = current * (1 + blended_ret_1m/100.0)
    pred_price_1y = current * (1 + blended_ret_1y/100.0)

    return {
        "Company": name, "Ticker": tkr,
        "Current": round(current, 2),
        "Pred 1M": round(pred_price_1m, 2),
        "Pred 1Y": round(pred_price_1y, 2),
        "Ret 1M %": round(blended_ret_1m, 2),
        "Ret 1Y %": round(blended_ret_1y, 2),
        "Prob Up 1M": round(float(p_up_1m), 3),
        "Prob Up 1Y": round(float(p_up_1y), 3),
    }

# --------------- UI ---------------
companies = fetch_midcap100_tickers()
if not companies: st.stop()

col1, col2, col3 = st.columns([2,2,2])
with col1:
    mode = st.radio("Mode", ["Fast", "Full"], index=0,
                    help="Fast = ML only (quick). Full = Prophet (weekly) + ML blend (slower).")
with col2:
    limit = st.slider("Tickers to process", 5, min(100, len(companies)), min(20, len(companies)), step=5)
with col3:
    risk_adj = st.slider("🌍 Risk Adjustment (%)", -20, 20, 0)

if st.button("Run Screener"):
    rows = []
    progress = st.progress(0.0)

    # Small thread pool to parallelize without hammering APIs
    max_workers = 6 if mode == "Fast" else 4
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, nt in enumerate(companies[:limit], start=1):
            tasks.append(ex.submit(process_one, nt, mode, risk_adj))
        done = 0
        for fut in as_completed(tasks):
            res = fut.result()
            if res: rows.append(res)
            done += 1
            progress.progress(done / max(1, len(tasks)))

    if not rows:
        st.error("No results. Try Fast mode or fewer tickers.")
    else:
        out = pd.DataFrame(rows)

        # Ranks (ascending: 1 = best)
        out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
        out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
        out["Rank Prob 1M"] = out["Prob Up 1M"].rank(ascending=False, method="min").astype(int)
        out["Rank Prob 1Y"] = out["Prob Up 1Y"].rank(ascending=False, method="min").astype(int)
        out["Composite Rank"] = (
            out["Rank Ret 1M"] + out["Rank Ret 1Y"] + out["Rank Prob 1M"] + out["Rank Prob 1Y"]
        ) / 4.0
        out["Composite Rank"] = out["Composite Rank"].rank(ascending=True, method="min").astype(int)

        # Final column order
        out = out[[
            "Company","Ticker","Current","Pred 1M","Pred 1Y",
            "Ret 1M %","Ret 1Y %","Prob Up 1M","Prob Up 1Y","Composite Rank",
            "Rank Ret 1M","Rank Ret 1Y","Rank Prob 1M","Rank Prob 1Y"
        ]].sort_values("Composite Rank").reset_index(drop=True)

        st.success("✅ Results Ready")

        # Styling
        rank_cols = ["Rank Ret 1M","Rank Ret 1Y","Rank Prob 1M","Rank Prob 1Y","Composite Rank"]
        return_cols = ["Ret 1M %","Ret 1Y %"]

        def color_returns(v):
            if v > 0: return "color: green"
            if v < 0: return "color: red"
            return ""

        styled = (
            out.style
            .background_gradient(cmap="RdYlGn_r", subset=rank_cols)
            .applymap(color_returns, subset=return_cols)
        )
        st.dataframe(styled, use_container_width=True)

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name=f"midcap_screener_{mode.lower()}.csv", mime="text/csv")
