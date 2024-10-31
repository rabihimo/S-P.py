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
    from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Define the top 3 S&P 500 companies
tickers = ['AAPL', 'MSFT', 'GOOGL']  # Adjust tickers as needed

# Download historical data
data = yf.download(tickers, period="5y")  # Adjust the period as needed

# Extract adjusted closing prices
adj_close = data['Adj Close']

# Ensure there is no missing data
adj_close.dropna(inplace=True)

# Plot the adjusted closing prices
plt.figure(figsize=(14, 7))
for ticker in tickers:
    plt.plot(adj_close.index, adj_close[ticker], label=ticker)

plt.xlabel('Date')
plt.ylabel('Adjusted Closing Price')
plt.title('Adjusted Closing Price of Top 3 S&P 500 Companies')
plt.legend()
plt.grid(True)
plt.show()

# Predictive modeling for AAPL (you can repeat for other tickers)
X = np.arange(len(adj_close['AAPL'])).reshape(-1, 1)  # Use index as predictor
y = adj_close['AAPL'].values  # Make sure this is a 1D array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Plot the predictions against actual values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Actual Prices', color='blue')
plt.plot(X_test, predictions, color='red', label='Predicted Prices')

plt.xlabel('Time Index')
plt.ylabel('Adjusted Close Price')
plt.title('Linear Regression Prediction for AAPL')
plt.legend()
plt.grid(True)
plt.show()

print("Note: This is a very basic linear regression example. For more accurate predictions, consider using more sophisticated models, adding more features, and handling time series data appropriately.")
