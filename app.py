import yfinance as yf
from prophet import Prophet
import pandas as pd
import streamlit as st
import math
from io import StringIO

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor & Midcap Screener")

# ---------- Helpers ----------
def normal_cdf(x):
    # CDF of standard normal using erf (no scipy needed)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, years: int = 5) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Flatten multiindex if returned
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.reset_index(inplace=True)
    # Choose price column robustly
    price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if price_col is None:
        return pd.DataFrame()
    df = df.dropna(subset=[price_col])
    df.rename(columns={"Date": "ds", price_col: "y"}, inplace=True)
    return df[["ds", "y"]]

@st.cache_data(show_spinner=False)
def fit_and_forecast(df: pd.DataFrame, horizon_days: int = 365) -> pd.DataFrame:
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon_days)
    fc = m.predict(future)
    return fc

def prob_above_current(fc_row, current_price):
    # Use Prophet's yhat +/- interval to estimate sigma, assume normal around yhat
    yhat = fc_row["yhat"]
    ylow = fc_row.get("yhat_lower", float("nan"))
    yhigh = fc_row.get("yhat_upper", float("nan"))
    # If intervals missing, fall back to zero-variance (prob ~ 1 if above, else 0.5 if equal, else 0)
    if pd.isna(ylow) or pd.isna(yhigh) or (yhigh <= ylow):
        return 1.0 if yhat > current_price else (0.5 if abs(yhat - current_price) < 1e-9 else 0.0)
    sigma = (yhigh - ylow) / (2.0 * 1.96)
    if sigma <= 0:
        return 1.0 if yhat > current_price else (0.5 if abs(yhat - current_price) < 1e-9 else 0.0)
    z = (yhat - current_price) / sigma
    return 1.0 - normal_cdf(-z)  # = normal SF of (current - mean)/sigma

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🔍 Single Ticker", "🧮 Nifty Midcap-100 Screener"])

with tab1:
    st.subheader("Single Ticker Forecast")
    ticker = st.text_input("Enter Stock Symbol (e.g. AAPL, RELIANCE.NS, INFY.NS)", "AAPL")
    if ticker:
        data = yf.download(ticker, period="5y", auto_adjust=False, progress=False)
        if data.empty:
            st.error("⚠️ No data found. Try another ticker (e.g., RELIANCE.NS).")
        else:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] for c in data.columns]
            data.reset_index(inplace=True)
            price_col = "Adj Close" if "Adj Close" in data.columns else ("Close" if "Close" in data.columns else None)
            if price_col is None:
                st.error("Price columns missing for this ticker.")
            else:
                st.line_chart(data.set_index("Date")[price_col])

                df = data[["Date", price_col]].rename(columns={"Date": "ds", price_col: "y"})
                m = Prophet(daily_seasonality=True)
                m.fit(df)
                fc = m.predict(m.make_future_dataframe(periods=365))

                st.subheader("🔮 Forecast")
                fig1 = m.plot(fc)
                st.pyplot(fig1)

                one_month = fc.iloc[-30]["yhat"]
                one_year = fc.iloc[-1]["yhat"]
                st.subheader("📌 Predictions")
                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("Current", f"{df['y'].iloc[-1]:.2f}")
                with colB:
                    st.metric("1-Month", f"{one_month:.2f}")
                with colC:
                    st.metric("1-Year", f"{one_year:.2f}")

with tab2:
    st.subheader("Nifty Midcap-100 Screener (Predicted ↑ vs Today)")
    st.caption("Tip: Start with a smaller subset for speed, then scale up.")

    # 🔧 Editable list of NSE Midcap-100 tickers (suffix .NS). Paste/modify as needed.
    default_list = """ABBOTINDIA.NS
ALKEM.NS
ASHOKLEY.NS
AUBANK.NS
AUROPHARMA.NS
BALKRISIND.NS
BEL.NS
BERGEPAINT.NS
BHEL.NS
CANFINHOME.NS
CUMMINSIND.NS
DALBHARAT.NS
DEEPAKNTR.NS
FEDERALBNK.NS
GODREJPROP.NS
HAVELLS.NS
HINDZINC.NS
IDFCFIRSTB.NS
INDHOTEL.NS
INDIAMART.NS
IPCALAB.NS
JUBLFOOD.NS
L&TFH.NS
LUPIN.NS
MANKIND.NS
MUTHOOTFIN.NS
NMDC.NS
OBEROIRLTY.NS
PAGEIND.NS
PIDILITIND.NS
PERSISTENT.NS
PFIZER.NS
PIDILITIND.NS
POLYCAB.NS
RECLTD.NS
SAIL.NS
SHREECEM.NS
SRF.NS
SUNTV.NS
TATAELXSI.NS
TATAPOWER.NS
TATACHEM.NS
TRENT.NS
TVSMOTOR.NS
UBL.NS
VOLTAS.NS
ZFCVINDIA.NS
ZYDUSLIFE.NS
"""
    tickers_txt = st.text_area(
        "Paste Midcap-100 tickers here (one per line). Suffix with .NS for NSE.",
        value=default_list,
        height=180
    )
    tickers = [t.strip() for t in tickers_txt.splitlines() if t.strip()]

    limit = st.slider("How many tickers to process (caching on)", min_value=5, max_value=min(100, len(tickers)), value=min(25, len(tickers)), step=5)
    run_btn = st.button("Run Screener")

    if run_btn and tickers:
        rows = []
        progress = st.progress(0)
        for i, tkr in enumerate(tickers[:limit], start=1):
            try:
                df = fetch_history(tkr, years=5)
                if df.empty or len(df) < 200:
                    continue
                current = float(df["y"].iloc[-1])

                fc = fit_and_forecast(df, horizon_days=365)
                # Target rows: last 30th from end (≈ +30d), last (≈ +365d)
                row_30 = fc.iloc[-30]
                row_365 = fc.iloc[-1]

                pred_30 = float(row_30["yhat"])
                pred_365 = float(row_365["yhat"])

                ret_30 = (pred_30 - current) / current * 100.0
                ret_365 = (pred_365 - current) / current * 100.0

                p_30 = prob_above_current(row_30, current)
                p_365 = prob_above_current(row_365, current)

                rows.append({
                    "Ticker": tkr,
                    "Current": round(current, 2),
                    "Pred 1M": round(pred_30, 2),
                    "Pred 1Y": round(pred_365, 2),
                    "Ret 1M %": round(ret_30, 2),
                    "Ret 1Y %": round(ret_365, 2),
                    "Prob Up 1M": round(p_30, 3),
                    "Prob Up 1Y": round(p_365, 3),
                })
            except Exception:
                # Skip any problematic ticker silently to keep the run smooth
                pass
            progress.progress(i / max(1, min(limit, len(tickers))))

        if not rows:
            st.error("No results. Try increasing the limit or adjusting tickers.")
        else:
            out = pd.DataFrame(rows)

            # Ranks (higher return => better rank, higher prob => better rank)
            out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
            out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
            out["Rank Prob 1M"] = out["Prob Up 1M"].rank(ascending=False, method="min").astype(int)
            out["Rank Prob 1Y"] = out["Prob Up 1Y"].rank(ascending=False, method="min").astype(int)

            # Composite rank example (average of four ranks)
            out["Composite Rank"] = (
                out["Rank Ret 1M"] + out["Rank Ret 1Y"] + out["Rank Prob 1M"] + out["Rank Prob 1Y"]
            ) / 4.0
            out["Composite Rank"] = out["Composite Rank"].rank(ascending=True, method="min").astype(int)

            # Sort by Composite Rank
            out = out.sort_values(["Composite Rank", "Rank Ret 1Y", "Rank Prob 1Y"]).reset_index(drop=True)

            st.success("Done! Showing ranked results.")
            st.dataframe(out, use_container_width=True)

            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results (CSV)", data=csv, file_name="midcap_screener.csv", mime="text/csv")

# end
