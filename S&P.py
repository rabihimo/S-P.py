import streamlit as st
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objs as go
import yfinance as yf
import pandas as pd

# Specify title and logo for the webpage.
st.set_page_config(layout="wide", page_title="WebApp_Demo")

st.sidebar.title("Input")
symbol = st.sidebar.text_input('Please enter the stock symbol: ', 'NVDA').upper()

col1, col2 = st.sidebar.columns(2, gap="medium")
with col1:
    sdate = st.date_input('Start Date', value=datetime.date(2024, 1, 1))
with col2:
    edate = st.date_input('End Date', value=datetime.date.today())

st.title(f"{symbol}")

try:
    stock = yf.Ticker(symbol)
    st.write(f"# Sector : {stock.info['sector']}")
    st.write(f"# Company Beta : {stock.info['beta']}")

    data = yf.download(symbol, start=sdate, end=edate)
    if data.empty:
        st.error(f"No data found for symbol '{symbol}' within the specified date range.")
    else:
        st.line_chart(data['Close'], x_label="Date", y_label="Close")

    # S&P 500 comparison (Day-by-Day Fluctuation)
    st.header("S&P 500 Top 5 Comparison (Day-by-Day Fluctuation)") # This and subsequent lines should be indented within the 'try' block
    top_5_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA']  # Example, replace with dynamic data

    top_5_data = {}
    for sym in top_5_symbols:
        try:
            top_5_data[sym] = yf.download(sym, start=sdate, end=edate)['Close']
        except Exception as e:
            st.warning(f"Could not retrieve data for {sym}: {e}")

    if top_5_data:
        comparison_df = pd.DataFrame(top_5_data)
        
        # Calculate daily percentage change
        daily_returns = comparison_df.pct_change() * 100

        st.line_chart(daily_returns)
    else:
        st.warning("Could not retrieve data for the S&P 500 top 5 comparison.")

except Exception as e:
    st.error(f"An error occurred: {e}")
