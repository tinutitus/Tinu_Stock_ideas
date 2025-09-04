import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
import streamlit as st
import math
import requests
from io import StringIO

# ML (lightweight)
from sklearn.linear_model import Ridge

st.set_page_config(page_title="Midcap-100 Screener", layout="wide")
st.title("🧮 Nifty Midcap-100 Screener — Blended Predictions (Prophet + ML)")

# ---------- Utils ----------
def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@st.cache_data(ttl=86400)
def fetch_midcap100_tickers():
    """Fetch Midcap 100 list with a fast timeout, fallback to full static list."""
    url = "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        tickers = df["Symbol"].astype(str).str.strip().apply(lambda s: f"{s}.NS").tolist()
        names = df["Company Name"].astype(str).str.strip().tolist()
        return list(zip(names, tickers))
    except Exception:
        # Full static fallback (100 common Midcap tickers; update as needed)
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
        return [(sym, f"{sym}.NS") for sym in fallback_symbols]

@st.cache_data(show_spinner=False)
def fetch_history(tkr: str, years: int = 5) -> pd.DataFrame:
    """Fetch historical prices for a ticker (cached). Returns df with ['ds','y']."""
    df = yf.download(tkr, period=f"{years}y", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.reset_index(inplace=True)
    price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if not price_col:
        return pd.DataFrame()
    df.rename(columns={"Date": "ds", price_col: "y"}, inplace=True)
    df = df[["ds", "y"]].dropna()
    return df

@st.cache_data(show_spinner=False)
def prophet_forecast(df: pd.DataFrame, horizon_days: int = 365) -> pd.DataFrame:
    """Fit Prophet and return forecast df."""
    m = Prophet(daily_seasonality=True)
    m.fit(df.rename(columns={"ds": "ds", "y": "y"}))
    future = m.make_future_dataframe(periods=horizon_days)
    fc = m.predict(future)
    return fc

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lightweight technical features for ML from price series."""
    df = df.copy()
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

    # Targets (forward returns)
    df["target_30d"]  = df["y"].shift(-30) / df["y"] - 1.0
    df["target_365d"] = df["y"].shift(-365) / df["y"] - 1.0

    # Drop early NaNs
    df = df.dropna().reset_index(drop=True)
    return df

def train_predict_ml(df: pd.DataFrame):
    """
    Train two Ridge models to predict 30d and 365d forward returns from features.
    Returns (ml_ret_30, ml_ret_365) as percentages.
    """
    if len(df) < 400:
        # Not enough history to make a decent ML fit
        return None, None

    feats = ["ret_5d", "ret_21d", "ret_63d", "ret_252d", "vol_21d", "rsi_14"]
    df_feat = make_features(df)
    if df_feat.empty:
        return None, None

    X = df_feat[feats]
    y30 = df_feat["target_30d"]
    y365 = df_feat["target_365d"]

    # Use the last available row as "current" features
    x_curr = X.iloc[[-1]]

    # Simple regularized linear model (fast, stable on Streamlit Cloud)
    try:
        m30 = Ridge(alpha=1.0).fit(X.iloc[:-1], y30.iloc[:-1])
        m365 = Ridge(alpha=1.0).fit(X.iloc[:-1], y365.iloc[:-1])
        pred30 = float(m30.predict(x_curr)[0]) * 100.0
        pred365 = float(m365.predict(x_curr)[0]) * 100.0
        return pred30, pred365
    except Exception:
        return None, None

def prob_above_current_from_prophet(fc_row, current_price):
    """Approximate P(price > current) using Prophet interval as normal approx."""
    yhat = fc_row["yhat"]
    ylow = fc_row.get("yhat_lower", float("nan"))
    yhigh = fc_row.get("yhat_upper", float("nan"))
    if pd.isna(ylow) or pd.isna(yhigh) or (yhigh <= ylow):
        return 1.0 if yhat > current_price else (0.5 if abs(yhat - current_price) < 1e-9 else 0.0)
    sigma = (yhigh - ylow) / (2.0 * 1.96)
    if sigma <= 0:
        return 1.0 if yhat > current_price else (0.5 if abs(yhat - current_price) < 1e-9 else 0.0)
    z = (yhat - current_price) / sigma
    return 1.0 - normal_cdf(-z)

# ---------- UI ----------
companies = fetch_midcap100_tickers()
if not companies:
    st.stop()

st.subheader("📊 Ranked predictions (ascending: 1 = best)")

limit = st.slider("Tickers to process", 5, min(100, len(companies)), min(20, len(companies)), step=5)
risk_adj = st.slider("🌍 Geopolitical Risk Adjustment (%)", -20, 20, 0,
                     help="Applies to predicted prices/returns.")
run_btn = st.button("Run Screener")

if run_btn:
    rows = []
    progress = st.progress(0)

    for i, (name, tkr) in enumerate(companies[:limit], start=1):
        try:
            hist = fetch_history(tkr)
            if hist.empty or len(hist) < 260:  # at least ~1yr of data
                continue

            current = float(hist["y"].iloc[-1])

            # Prophet forecast (1M ~ 30 days, 1Y ~ 365 days)
            fc = prophet_forecast(hist, horizon_days=365)
            row_30, row_365 = fc.iloc[-30], fc.iloc[-1]
            prop_pred30 = float(row_30["yhat"])
            prop_pred365 = float(row_365["yhat"])

            # ML returns (%) from technical features
            ml_ret_30, ml_ret_365 = train_predict_ml(hist)

            # Convert Prophet to returns (%)
            prop_ret_30 = (prop_pred30 - current) / current * 100.0
            prop_ret_365 = (prop_pred365 - current) / current * 100.0

            # Blend (if ML is available; else fallback to Prophet only)
            if ml_ret_30 is not None and ml_ret_365 is not None:
                blended_ret_30 = 0.5 * prop_ret_30 + 0.5 * ml_ret_30
                blended_ret_365 = 0.5 * prop_ret_365 + 0.5 * ml_ret_365
            else:
                blended_ret_30 = prop_ret_30
                blended_ret_365 = prop_ret_365

            # Apply geopolitical adjustment to returns
            blended_ret_30 *= (1 + risk_adj / 100.0)
            blended_ret_365 *= (1 + risk_adj / 100.0)

            # Convert blended returns back to predicted prices
            pred_price_1m = current * (1.0 + blended_ret_30 / 100.0)
            pred_price_1y = current * (1.0 + blended_ret_365 / 100.0)

            # Probabilities from Prophet intervals (not blended; serves as risk signal)
            p_up_1m = prob_above_current_from_prophet(row_30, current)
            p_up_1y = prob_above_current_from_prophet(row_365, current)

            rows.append({
                "Company": name,
                "Ticker": tkr,
                "Current": round(current, 2),
                "Pred 1M": round(pred_price_1m, 2),
                "Pred 1Y": round(pred_price_1y, 2),
                "Ret 1M %": round(blended_ret_30, 2),
                "Ret 1Y %": round(blended_ret_365, 2),
                "Prob Up 1M": round(p_up_1m, 3),
                "Prob Up 1Y": round(p_up_1y, 3),
            })
        except Exception:
            # Skip silently to keep the batch running
            pass

        progress.progress(i / max(1, min(limit, len(companies))))

    if not rows:
        st.error("No results. Try more tickers.")
    else:
        out = pd.DataFrame(rows)

        # Ranks (ascending: 1 = best)
        out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
        out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
        out["Rank Prob 1M"] = out["Prob Up 1M"].rank(ascending=False, method="min").astype(int)
        out["Rank Prob 1Y"] = out["Prob Up 1Y"].rank(ascending=False, method="min").astype(int)

        # Composite = average of the four ranks (blended returns + probabilities)
        out["Composite Rank"] = (
            out["Rank Ret 1M"] + out["Rank Ret 1Y"] + out["Rank Prob 1M"] + out["Rank Prob 1Y"]
        ) / 4.0
        out["Composite Rank"] = out["Composite Rank"].rank(ascending=True, method="min").astype(int)

        # Final column order (as requested)
        out = out[[
            "Company","Ticker","Current","Pred 1M","Pred 1Y","Ret 1M %","Ret 1Y %","Prob Up 1M","Prob Up 1Y","Composite Rank",
            "Rank Ret 1M","Rank Ret 1Y","Rank Prob 1M","Rank Prob 1Y"  # keep detail ranks at the end
        ]].sort_values("Composite Rank").reset_index(drop=True)

        st.success("✅ Results Ready")

        # Styling
        rank_cols = ["Rank Ret 1M","Rank Ret 1Y","Rank Prob 1M","Rank Prob 1Y","Composite Rank"]
        return_cols = ["Ret 1M %","Ret 1Y %"]

        def color_returns(val):
            if val > 0: return "color: green"
            if val < 0: return "color: red"
            return ""

        styled = (
            out.style
            .background_gradient(cmap="RdYlGn_r", subset=rank_cols)  # green best → red worst
            .applymap(color_returns, subset=return_cols)             # green gains, red losses
        )

        st.dataframe(styled, use_container_width=True)

        # CSV Download
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results (CSV)", data=csv, file_name="midcap_screener_blended.csv", mime="text/csv")
