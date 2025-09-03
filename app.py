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
    # If intervals missing, fall back to zero-variance (prob ~ 1 if above, else 0.5 if equal,
