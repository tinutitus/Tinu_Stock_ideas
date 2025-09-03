import yfinance as yf
from prophet import Prophet
import pandas as pd
import streamlit as st
import math

st.set_page_config(page_title="Midcap-100 Screener", layout="wide")
st.title("🧮 Nifty Midcap-100 Screener (Final Hybrid Version)")

# ---------- Helpers ----------
def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@st.cache_data(ttl=86400)
def fetch_midcap100_tickers():
    """Try to fetch Midcap 100 from NSE Indices CSV, fallback to static list if it fails."""
    url = "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv"
    try:
        df = pd.read_csv(url)
        tickers = df["Symbol"].astype(str).str.strip().apply(lambda s: f"{s}.NS").tolist()
        names = df["Company Name"].astype(str).str.strip().tolist()
        return list(zip(names, tickers))
    except Exception:
        # ---- full static fallback list (100 tickers) ----
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
def fetch_history(tkr, years=5):
    """Fetch historical prices for a ticker (cached)."""
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
    """Fit Prophet model and forecast future prices (cached)."""
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

st.subheader("📊 Nifty Midcap-100 Predictions (Hybrid Fetch)")

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
