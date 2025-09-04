import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import requests
from io import StringIO
from typing import List, Tuple
from datetime import timedelta

# light ML + ETS (fast)
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Nifty Midcap-100 Screener — Fast + Accuracy", layout="wide")
st.title("⚡ Nifty Midcap-100 Screener — Fast (Lite) + Accuracy Mode (Top N)")

# ------------------------- Helpers -------------------------
@st.cache_data(ttl=86400)
def fetch_midcap100_tickers() -> List[Tuple[str,str]]:
    """Try the CSV (5s timeout), else fallback to full static list. Returns [(Company, Ticker.NS), ...]."""
    url = "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        names = df["Company Name"].astype(str).str.strip().tolist()
        tks = df["Symbol"].astype(str).str.strip().apply(lambda s: f"{s}.NS").tolist()
        return list(zip(names, tks))
    except Exception:
        fallback = [
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
        return [(s, f"{s}.NS") for s in fallback]

@st.cache_data(show_spinner=False)
def batch_history(tickers: List[str], years: int = 3) -> pd.DataFrame:
    """Batch download for many tickers in ONE call (fast)."""
    data = yf.download(tickers, period=f"{years}y", auto_adjust=False, progress=False,
                       threads=True, group_by='ticker')
    return data

def _compute_signals(series: pd.Series):
    """Fast signals for Lite pass."""
    s = series.dropna()
    if s.size < 80: return None
    y = s.values
    current = float(y[-1])

    def mom(days):
        if s.size <= days: return np.nan
        return (y[-1] / y[-days] - 1.0) * 100.0

    m21  = mom(21)    # ~1M
    m63  = mom(63)    # ~3M
    m252 = mom(252)   # ~1Y if available

    vol21 = s.pct_change().rolling(21).std().iloc[-1]
    if np.isnan(vol21): vol21 = s.pct_change().std()

    # Heuristic returns, volatility-penalized
    ret_1m_est = np.clip(0.6*m21 + 0.4*m63 - 100*vol21, -50, 50)
    m252_proxy = m252 if not np.isnan(m252) else (m63*4 if not np.isnan(m63) else 0.0)
    ret_1y_est = np.clip(0.3*m63 + 0.7*m252_proxy - 150*vol21, -80, 120)

    # Prob proxies
    p1m = float(np.clip(0.5 + (m21/100.0) - (vol21*2), 0.05, 0.95))
    p1y = float(np.clip(0.5 + ((m252 if not np.isnan(m252) else m63)/300.0) - (vol21*1.5), 0.05, 0.95))

    return dict(current=current, ret1m=ret_1m_est, ret1y=ret_1y_est, p1m=p1m, p1y=p1y)

def build_lite_table(companies: List[Tuple[str,str]], limit: int, risk_adj: float) -> pd.DataFrame:
    subset = companies[:limit]
    names_map = {tkr: name for name, tkr in subset}
    tickers = [tkr for _, tkr in subset]

    data = batch_history(tickers, years=3)
    rows = []

    if isinstance(data.columns, pd.MultiIndex):
        for tkr in tickers:
            cols = list(data[tkr].columns)
            price_col = "Adj Close" if "Adj Close" in cols else ("Close" if "Close" in cols else None)
            if not price_col: continue
            sig = _compute_signals(data[tkr][price_col])
            if not sig: continue
            # risk adjust returns
            ret1m = sig["ret1m"] * (1 + risk_adj/100.0)
            ret1y = sig["ret1y"] * (1 + risk_adj/100.0)
            pred1m = sig["current"] * (1 + ret1m/100.0)
            pred1y = sig["current"] * (1 + ret1y/100.0)
            rows.append({
                "Company": names_map.get(tkr, tkr.replace(".NS","")),
                "Ticker": tkr,
                "Current": round(sig["current"], 2),
                "Pred 1M": round(pred1m, 2),
                "Pred 1Y": round(pred1y, 2),
                "Ret 1M %": round(ret1m, 2),
                "Ret 1Y %": round(ret1y, 2),
                "Prob Up 1M": round(sig["p1m"], 3),
                "Prob Up 1Y": round(sig["p1y"], 3),
            })
    else:
        # Single ticker case (rare in our UI, but safe-guard)
        cols = list(data.columns)
        price_col = "Adj Close" if "Adj Close" in cols else ("Close" if "Close" in cols else None)
        if price_col:
            tkr = tickers[0]
            sig = _compute_signals(data[price_col])
            if sig:
                ret1m = sig["ret1m"] * (1 + risk_adj/100.0)
                ret1y = sig["ret1y"] * (1 + risk_adj/100.0)
                pred1m = sig["current"] * (1 + ret1m/100.0)
                pred1y = sig["current"] * (1 + ret1y/100.0)
                rows.append({
                    "Company": names_map.get(tkr, tkr.replace(".NS","")),
                    "Ticker": tkr,
                    "Current": round(sig["current"], 2),
                    "Pred 1M": round(pred1m, 2),
                    "Pred 1Y": round(pred1y, 2),
                    "Ret 1M %": round(ret1m, 2),
                    "Ret 1Y %": round(ret1y, 2),
                    "Prob Up 1M": round(sig["p1m"], 3),
                    "Prob Up 1Y": round(sig["p1y"], 3),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Ranks (ascending: 1 = best)
    out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
    out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
    out["Rank Prob 1M"] = out["Prob Up 1M"].rank(ascending=False, method="min").astype(int)
    out["Rank Prob 1Y"] = out["Prob Up 1Y"].rank(ascending=False, method="min").astype(int)
    out["Composite Rank"] = (
        out["Rank Ret 1M"] + out["Rank Ret 1Y"] + out["Rank Prob 1M"] + out["Rank Prob 1Y"]
    ) / 4.0
    out["Composite Rank"] = out["Composite Rank"].rank(ascending=True, method="min").astype(int)

    return out.sort_values("Composite Rank").reset_index(drop=True)

# ---------------------- Accuracy mode (Top N) ----------------------
def ridge_features(series: pd.Series) -> pd.DataFrame:
    s = series.dropna().rename("y")
    df = pd.DataFrame({"y": s})
    df["ret_1d"]   = df["y"].pct_change(1)
    df["ret_5d"]   = df["y"].pct_change(5)
    df["ret_21d"]  = df["y"].pct_change(21)
    df["ret_63d"]  = df["y"].pct_change(63)
    df["ret_252d"] = df["y"].pct_change(252)
    df["vol_21d"]  = df["ret_1d"].rolling(21).std()
    # RSI(14)
    delta = df["y"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi_14"] = 100 - (100 / (1 + rs))
    # Targets
    df["tgt_21"] = df["y"].shift(-21)/df["y"] - 1.0
    df["tgt_252"] = df["y"].shift(-252)/df["y"] - 1.0
    return df.dropna().reset_index(drop=True)

def ridge_predict(series: pd.Series):
    df = ridge_features(series)
    feats = ["ret_5d","ret_21d","ret_63d","ret_252d","vol_21d","rsi_14"]
    if df.shape[0] < 200:  # not enough data
        return None, None, None
    X = df[feats]
    y21, y252 = df["tgt_21"], df["tgt_252"]
    x_curr = X.iloc[[-1]]
    try:
        m21 = Ridge(alpha=1.0).fit(X.iloc[:-1], y21.iloc[:-1])
        m252 = Ridge(alpha=1.0).fit(X.iloc[:-1], y252.iloc[:-1])
        pred21 = float(m21.predict(x_curr)[0])*100.0
        pred252 = float(m252.predict(x_curr)[0])*100.0
        return pred21, pred252, (m21, m252)
    except Exception:
        return None, None, None

def ets_predict(series: pd.Series):
    """Holt-Winters ETS on daily prices with weekly seasonality (5 trading days)."""
    s = series.dropna()
    if s.size < 150:
        return None, None, None
    try:
        model = ExponentialSmoothing(s, trend="add", damped_trend=True, seasonal="add", seasonal_periods=5)
        fit = model.fit(optimized=True, use_brute=False)
        fcast = fit.forecast(252)  # ~1y trading days
        curr = float(s.iloc[-1])
        pred_1m = float(fcast.iloc[21-1]) if len(fcast) >= 21 else float(fcast.iloc[-1])
        pred_1y = float(fcast.iloc[252-1]) if len(fcast) >= 252 else float(fcast.iloc[-1])
        ret_1m = (pred_1m - curr)/curr*100.0
        ret_1y = (pred_1y - curr)/curr*100.0
        return ret_1m, ret_1y, fit
    except Exception:
        return None, None, None

def backtest_mape_21(series: pd.Series, folds: int = 3):
    """Tiny rolling backtest for 1M horizon MAPE using ETS; lightweight."""
    s = series.dropna()
    if s.size < 200: return None
    step = max(10, (len(s)-200)//folds)
    errors = []
    for i in range(folds):
        cut = len(s) - (folds-i)*step - 21
        if cut < 100: break
        train = s.iloc[:cut]
        true = s.iloc[cut+21-1] if cut+21-1 < len(s) else s.iloc[-1]
        try:
            model = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=5)
            fit = model.fit(optimized=True)
            pred = float(fit.forecast(21).iloc[-1])
            mape = abs((true - pred)/true)*100.0 if true != 0 else 0.0
            errors.append(mape)
        except Exception:
            continue
    if not errors: return None
    return round(float(np.mean(errors)), 2)

def improve_topN(lite_df: pd.DataFrame, companies: List[Tuple[str,str]], top_n: int, risk_adj: float) -> pd.DataFrame:
    # Map ticker -> price series from a fresh batch (to ensure we have columns handy)
    tickers = lite_df["Ticker"].tolist()[:max(top_n, 1)]
    data = batch_history(tickers, years=3)

    enhanced_rows = []
    for tkr in tickers:
        # Extract price series
        if isinstance(data.columns, pd.MultiIndex):
            cols = list(data[tkr].columns)
            price_col = "Adj Close" if "Adj Close" in cols else ("Close" if "Close" in cols else None)
            if not price_col: continue
            series = data[tkr][price_col]
        else:
            # Single case
            cols = list(data.columns)
            price_col = "Adj Close" if "Adj Close" in cols else ("Close" if "Close" in cols else None)
            if not price_col: continue
            series = data[price_col]

        series = series.dropna()
        if series.size < 150:  # need some depth for ETS
            continue

        curr = float(series.iloc[-1])

        # ETS & Ridge predictions (returns %)
        ets_1m, ets_1y, _ = ets_predict(series)
        ridge_1m, ridge_1y, _ = ridge_predict(series)

        # If one is missing, fallback to the other; else blend 50/50
        if ets_1m is None and ridge_1m is None:
            continue
        if ets_1m is None:
            blend_1m = ridge_1m
        elif ridge_1m is None:
            blend_1m = ets_1m
        else:
            blend_1m = 0.5*ets_1m + 0.5*ridge_1m

        if ets_1y is None and ridge_1y is None:
            continue
        if ets_1y is None:
            blend_1y = ridge_1y
        elif ridge_1y is None:
            blend_1y = ets_1y
        else:
            blend_1y = 0.5*ets_1y + 0.5*ridge_1y

        # risk adjustment
        blend_1m *= (1 + risk_adj/100.0)
        blend_1y *= (1 + risk_adj/100.0)

        pred_1m = curr * (1 + blend_1m/100.0)
        pred_1y = curr * (1 + blend_1y/100.0)

        # quick probability proxies (reuse Lite idea)
        vol21 = series.pct_change().rolling(21).std().iloc[-1]
        if np.isnan(vol21): vol21 = series.pct_change().std()
        m21 = (series.iloc[-1]/series.iloc[-21]-1.0)*100.0 if len(series)>21 else 0.0
        m252 = (series.iloc[-1]/series.iloc[-252]-1.0)*100.0 if len(series)>252 else m21*4.0
        p1m = float(np.clip(0.5 + (m21/100.0) - (vol21*2), 0.05, 0.95))
        p1y = float(np.clip(0.5 + (m252/300.0) - (vol21*1.5), 0.05, 0.95))

        # backtest error badge (MAPE 1M)
        mape_1m = backtest_mape_21(series, folds=3)

        enhanced_rows.append({
            "Ticker": tkr,
            "Current": round(curr, 2),
            "Pred 1M": round(pred_1m, 2),
            "Pred 1Y": round(pred_1y, 2),
            "Ret 1M %": round(blend_1m, 2),
            "Ret 1Y %": round(blend_1y, 2),
            "Prob Up 1M": round(p1m, 3),
            "Prob Up 1Y": round(p1y, 3),
            "MAPE 1M %": mape_1m if mape_1m is not None else None,
        })

    enhanced_df = pd.DataFrame(enhanced_rows)
    if enhanced_df.empty:
        return lite_df

    # Merge back into lite table for those tickers (overwrite refined columns)
    merged = lite_df.copy()
    refined_cols = ["Pred 1M","Pred 1Y","Ret 1M %","Ret 1Y %","Prob Up 1M","Prob Up 1Y"]
    merged = merged.merge(enhanced_df[["Ticker"] + refined_cols + ["MAPE 1M %"]],
                          on="Ticker", how="left", suffixes=("", "_ref"))

    for c in refined_cols:
        merged[c] = merged[c+"_ref"].combine_first(merged[c])
        merged.drop(columns=[c+"_ref"], inplace=True)

    # Recompute ranks (ascending: 1 = best)
    merged["Rank Ret 1M"] = merged["Ret 1M %"].rank(ascending=False, method="min").astype(int)
    merged["Rank Ret 1Y"] = merged["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
    merged["Rank Prob 1M"] = merged["Prob Up 1M"].rank(ascending=False, method="min").astype(int)
    merged["Rank Prob 1Y"] = merged["Prob Up 1Y"].rank(ascending=False, method="min").astype(int)
    merged["Composite Rank"] = (
        merged["Rank Ret 1M"] + merged["Rank Ret 1Y"] + merged["Rank Prob 1M"] + merged["Rank Prob 1Y"]
    ) / 4.0
    merged["Composite Rank"] = merged["Composite Rank"].rank(ascending=True, method="min").astype(int)

    # Keep order & show MAPE if available
    cols = ["Company","Ticker","Current","Pred 1M","Pred 1Y","Ret 1M %","Ret 1Y %","Prob Up 1M","Prob Up 1Y","Composite Rank",
            "Rank Ret 1M","Rank Ret 1Y","Rank Prob 1M","Rank Prob 1Y","MAPE 1M %"]
    merged = merged[cols].sort_values("Composite Rank").reset_index(drop=True)
    return merged

# ------------------------- UI -------------------------
companies = fetch_midcap100_tickers()
if not companies:
    st.stop()

c1, c2, c3 = st.columns([2,2,2])
with c1:
    limit = st.slider("Tickers to process (Lite)", 5, min(100, len(companies)), min(30, len(companies)), step=5)
with c2:
    risk_adj = st.slider("🌍 Risk Adjustment (%)", -20, 20, 0)
with c3:
    top_n = st.slider("Top N for Accuracy mode", 5, 30, 10, step=1, help="We will refine only these.")

run = st.button("Run (Lite)")
if run:
    lite = build_lite_table(companies, limit, risk_adj)
    if lite.empty:
        st.error("No results in Lite pass. Try more tickers.")
    else:
        st.success("✅ Lite results ready.")
        rank_cols = ["Rank Ret 1M","Rank Ret 1Y","Rank Prob 1M","Rank Prob 1Y","Composite Rank"]
        return_cols = ["Ret 1M %","Ret 1Y %"]
        def color_returns(v):
            if v > 0: return "color: green"
            if v < 0: return "color: red"
            return ""
        styled = (lite.style.background_gradient(cmap="RdYlGn_r", subset=rank_cols)
                        .applymap(color_returns, subset=return_cols))
        st.dataframe(styled, use_container_width=True)
        csv = lite.to_csv(index=False).encode()
        st.download_button("Download CSV (Lite)", csv, "midcap_screener_lite.csv", "text/csv")

        st.divider()
        if st.button(f"🔬 Improve Top {top_n} (Accuracy mode: ETS + Ridge blend)"):
            refined = improve_topN(lite.head(top_n), companies, top_n, risk_adj)
            st.success("✅ Accuracy mode complete (Top N re-ranked).")
            styled2 = (refined.style.background_gradient(cmap="RdYlGn_r", subset=rank_cols)
                                .applymap(color_returns, subset=return_cols))
            st.dataframe(styled2, use_container_width=True)
            csv2 = refined.to_csv(index=False).encode()
            st.download_button("Download CSV (Accuracy Top N)", csv2, "midcap_screener_accuracy.csv", "text/csv")
