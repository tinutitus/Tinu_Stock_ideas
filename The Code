import yfinance as yf
from prophet import Prophet
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.title("📈 Stock Price Predictor")

# User input
ticker = st.text_input("Enter Stock Symbol (e.g. AAPL, INFY.BO, RELIANCE.NS)", "AAPL")

if ticker:      
    # Fetch data
    data = yf.download(ticker, period="5y")
    data.reset_index(inplace=True)

    st.subheader("📊 Historical Stock Prices")
    st.line_chart(data[["Date", "Close"]].set_index("Date"))

    # Prepare data for Prophet
    df = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

    # Train model
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    # Future predictions
    future = model.make_future_dataframe(periods=365)  # 1 year ahead
    forecast = model.predict(future)

    # Plot forecast
    st.subheader("🔮 Forecast")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Show predictions for 1 month & 1 year
    one_month_price = forecast.iloc[-30]["yhat"]
    one_year_price = forecast.iloc[-1]["yhat"]

    st.metric("Predicted Price (1 Month)", f"${one_month_price:.2f}")
    st.metric("Predicted Price (1 Year)", f"${one_year_price:.2f}")
