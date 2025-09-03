import yfinance as yf
from prophet import Prophet
import pandas as pd
import streamlit as st
import math
import requests

st.set_page_config(page_title="Midcap-100 Screener", layout="wide")
st.title("🧮 Nifty Midcap-100 Screener (Dynamic from NSE API)")

# ---------- Helpers ----------
def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@st.cache_data(ttl=86400)
def fetch_midcap100_tickers():
    """Fetch Nifty Midcap 100 constituents dynamically from NSE API with session & headers."""
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MIDCAP%20100"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        session = requests.Session()
        # Load NSE homepage first to get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        # Now call API with cookies + headers
        r = session.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        rows = data.get("data", [])
        tickers = []
        for row in rows:
            symbol = row.get("symbol")
            name = row.get("identifier", symbol)
            if symbol:
                tickers.append((name, f"{symbol}.NS"))
        return tickers
    except Exception as e:
        st.error(f"⚠️ Failed to fetch Midcap-100 list: {e}")
        return []

@st.cache_data(show_spinner=False)
def fetch_history(tkr, years=5):
    """Fetch historical prices for a ticker."""
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
    return df[["ds", "y"]]

@st.cache_data(show_spinner=False)
def fit_and_forecast(df, days=365):
    """Fit Prophet model and forecast future prices."""
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=days)
    return m.predict(future)

def prob_above_current(fc_row, current):
    """Estimate probability that forecasted price > current."""
    yhat = fc_row["yhat"]
    ylow = fc_row.get("yhat_lower", float("nan"))
    yhigh = fc_row.get("yhat_upper", float("nan"))
    if pd.isna(ylow) or pd.isna(yhigh) or (yhigh <= ylow):
        return 1.0 if yhat > current else (0.5 if abs(yhat - current) < 1e-9 else 0.0)
    sigma = (yhigh - ylow) / (2.0 * 1.96)
    if sigma <= 0:
        return 1.0 if yhat > current else (0.5 if abs(yhat - current) < 1e-9 else 0.0)
    z = (yhat - current) / sigma
    return 1.0 - normal_cdf(-z)

# ---------- UI ----------
companies = fetch_midcap100_tickers()
if not companies:
    st.stop()

st.subheader("📊 Nifty Midcap-100 Predictions (auto-updated daily)")

limit = st.slider("Tickers to process", min_value=5, max_value=min(100, len(companies)), value=20, step=5)
risk_adj = st.slider("🌍 Geopolitical Risk Adjustment (%)", -20, 20, 0)
run_btn = st.button("Run Screener")

if run_btn:
    rows = []
    progress = st.progress(0)

    for i, (name, tkr) in enumerate(companies[:limit], start=1):
        try:
            df = fetch_history(tkr)
            if df.empty or len(df) < 200:
                continue
            current = df["y"].iloc[-1]
            fc = fit_and_forecast(df)
            row_30, row_365 = fc.iloc[-30], fc.iloc[-1]
            pred30, pred365 = row_30["yhat"], row_365["yhat"]

            # Apply geopolitical adjustment
            pred30 *= (1 + risk_adj / 100)
            pred365 *= (1 + risk_adj / 100)

            ret30 = (pred30 - current) / current * 100
            ret365 = (pred365 - current) / current * 100

            p30 = prob_above_current(row_30, current)
            p365 = prob_above_current(row_365, current)

            rows.append({
                "Company": name,
                "Ticker": tkr,
                "Current": round(current, 2),
                "Pred 1M": round(pred30, 2),
                "Pred 1Y": round(pred365, 2),
                "Ret 1M %": round(ret30, 2),
                "Ret 1Y %": round(ret365, 2),
                "Prob Up 1M": round(p30, 3),
                "Prob Up 1Y": round(p365, 3),
            })
        except Exception:
            pass
        progress.progress(i / limit)

    if not rows:
        st.error("No results. Try more tickers.")
    else:
        out = pd.DataFrame(rows)

        # ---------- Ranks (ascending: 1 = best) ----------
        out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
        out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
        out["Rank Prob 1M"] = out["Prob Up 1M"].rank(ascending=False, method="min").astype(int)
        out["Rank Prob 1Y"] = out["Prob Up 1Y"].rank(ascending=False, method="min").astype(int)

        out["Composite Rank"] = (
            out["Rank Ret 1M"] + out["Rank Ret 1Y"] + out["Rank Prob 1M"] + out["Rank Prob 1Y"]
        ) / 4.0
        out["Composite Rank"] = out["Composite Rank"].rank(ascending=True, method="min").astype(int)

        out = out.sort_values("Composite Rank").reset_index(drop=True)

        st.success("✅ Results Ready")

        # --- Styling ---
        rank_cols = ["Rank Ret 1M", "Rank Ret 1Y", "Rank Prob 1M", "Rank Prob 1Y", "Composite Rank"]
        return_cols = ["Ret 1M %", "Ret 1Y %"]

        def color_returns(val):
            if val > 0:
                return "color: green"
            elif val < 0:
                return "color: red"
            return ""

        styled = (
            out.style
            .background_gradient(cmap="RdYlGn_r", subset=rank_cols)
            .applymap(color_returns, subset=return_cols)
        )

        st.dataframe(styled, use_container_width=True)

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results (CSV)", data=csv, file_name="midcap_screener.csv", mime="text/csv")
