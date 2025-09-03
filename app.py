import yfinance as yf
from prophet import Prophet
import pandas as pd
import streamlit as st
import math

st.set_page_config(page_title="Stock Predictor & Midcap Screener", layout="wide")
st.title("📈 Stock Price Predictor & Midcap-100 Screener")

# ---------- Helpers ----------
def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, years: int = 5) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.reset_index(inplace=True)
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
    return m.predict(future)

def prob_above_current(fc_row, current_price):
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

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🔍 Single Ticker", "🧮 Nifty Midcap-100 Screener"])

# ---------- Tab 1: Single Ticker ----------
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
            if price_col:
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

# ---------- Tab 2: Midcap Screener ----------
with tab2:
    st.subheader("Nifty Midcap-100 Screener (Predicted ↑ vs Today)")

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
LUPIN.NS
MANKIND.NS
MUTHOOTFIN.NS
NMDC.NS
OBEROIRLTY.NS
PAGEIND.NS
PERSISTENT.NS
PFIZER.NS
POLYCAB.NS
RECLTD.NS
SAIL.NS
SRF.NS
SUNTV.NS
TATAELXSI.NS
TATAPOWER.NS
TRENT.NS
TVSMOTOR.NS
UBL.NS
VOLTAS.NS
ZYDUSLIFE.NS
"""
    tickers_txt = st.text_area(
        "Paste Midcap-100 tickers here (one per line, suffix .NS).",
        value=default_list,
        height=180
    )
    tickers = [t.strip() for t in tickers_txt.splitlines() if t.strip()]

    limit = st.slider("How many tickers to process", min_value=5, max_value=min(100, len(tickers)), value=min(20, len(tickers)), step=5)
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
                row_30, row_365 = fc.iloc[-30], fc.iloc[-1]
                pred_30, pred_365 = float(row_30["yhat"]), float(row_365["yhat"])
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
                pass
            progress.progress(i / max(1, min(limit, len(tickers))))

        if not rows:
            st.error("No results. Try more tickers.")
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

            out = out.sort_values(["Composite Rank", "Rank Ret 1Y", "Rank Prob 1Y"]).reset_index(drop=True)

            st.success("Done! Showing ranked results.")

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

            # Download CSV
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results (CSV)", data=csv, file_name="midcap_screener.csv", mime="text/csv")
