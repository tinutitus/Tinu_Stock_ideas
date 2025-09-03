import yfinance as yf
from prophet import Prophet
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.title("📈 Stock Price Predictor")

# User input
ticker = st.text_input("Enter Stock Symbol (e.g. AAPL, INFY.BO, RELIANCE.NS)", "AAPL")

if ticker:
    # Fetch last 5 years of data
    data = yf.download(ticker, period="5y")

    if data.empty:
        st.error("⚠️ No data found for this ticker. Try another symbol (e.g. AAPL, RELIANCE.NS).")
    else:
        # Ensure Date column exists
        data.reset_index(inplace=True)

        st.subheader("📊 Historical Stock Prices")
        st.line_chart(data.set_index("Date")["Adj Close"])

        # Prepare data for Prophet
        df = data[["Date", "Adj Close"]].rename(columns={"Date": "ds", "Adj Close": "y"})

        # Train Prophet model
        model = Prophet(daily_seasonality=True)
        model.fit(df)

        # Forecast for 1 year ahead
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        # Plot forecast
        st.subheader("🔮 Forecast Chart")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # Predicted values
        one_month_price = forecast.iloc[-30]["yhat"]
        one_year_price = forecast.iloc[-1]["yhat"]

        st.subheader("📌 Predictions")
        st.metric("Predicted Price (1 Month)", f"${one_month_price:.2f}")
        st.metric("Predicted Price (1 Year)", f"${one_year_price:.2f}")
