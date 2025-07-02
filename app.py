import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(page_title="📈 Stock Price Predictor with XGBoost", layout="wide")
st.title("📈 Stock Price Prediction App")
st.markdown("Predict Stock to get your profit")

# --- User Input ---
stock_symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE.BO, TCS.BO, INFY.NS):", "RELIANCE.BO")
start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.date_input("End Date", datetime.today())
days_to_predict = st.slider("Days to predict", 1, 30, 5)

# --- Load Stock Data ---
@st.cache_data
def load_data(symbol, start, end):
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end)
    df.dropna(inplace=True)
    return df

try:
    # Load historical data
    df = load_data(stock_symbol, start_date, end_date)
    ticker = yf.Ticker(stock_symbol)  # Used again for live section

    st.subheader("📊 Recent Historical Data")
    st.dataframe(df.tail(10).style.format({
        "Open": "₹{:.2f}", "Close": "₹{:.2f}", "High": "₹{:.2f}",
        "Low": "₹{:.2f}", "Volume": "{:.0f}"
    }))

    # --- Plotly Historical Chart ---
    st.subheader("📉 Historical Closing Price")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue')
    ))
    fig_hist.update_layout(
        title='Historical Closing Price',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Feature Engineering ---
    st.subheader("🤖 Predictions")

    df_feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_feat['Close_1d_ago'] = df_feat['Close'].shift(1)
    df_feat['Close_2d_ago'] = df_feat['Close'].shift(2)
    df_feat['MA7'] = df_feat['Close'].rolling(window=7).mean()
    df_feat['EMA10'] = df_feat['Close'].ewm(span=10, adjust=False).mean()
    df_feat['Returns'] = df_feat['Close'].pct_change()
    df_feat['Target'] = df_feat['Close'].shift(-days_to_predict)
    df_feat.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Volume', 'Close_1d_ago', 'Close_2d_ago', 'MA7', 'EMA10', 'Returns']
    X = df_feat[features]
    y = df_feat['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # --- Make Predictions ---
    latest_features = df_feat[features].tail(days_to_predict)
    latest_scaled = scaler.transform(latest_features)
    predictions = model.predict(latest_scaled)

    future_dates = [df.index[-1] + timedelta(days=i + 1) for i in range(days_to_predict)]
    predicted_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price (₹)': predictions})
    predicted_df.set_index('Date', inplace=True)

    st.subheader(f"📌 Predicted Prices for Next {days_to_predict} Days")
    st.dataframe(predicted_df.style.format("₹{:.2f}"))

    # --- Plotly Future Price Chart ---
    st.subheader("🔮 Future Price Chart")

    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    fig_future.add_trace(go.Scatter(
        x=predicted_df.index,
        y=predicted_df['Predicted Close Price (₹)'],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='green', dash='dash')
    ))
    fig_future.update_layout(
        title=f"Predicted Prices for Next {days_to_predict} Days",
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # --- Live Price + Intraday Chart ---
    st.subheader("📡 Live Stock Price")
    try:
        live_price = ticker.fast_info['last_price']
        st.success(f"Live Price: ₹{live_price:.2f}")

        st.subheader("📈 Live Updating Intraday Price Chart (Realistic)")
        chart_placeholder = st.empty()

        if st.button("Start Live Chart"):
            for _ in range(60):  # 5 mins @ 5s interval
                intraday_data = ticker.history(period="1d", interval="1m")
                if not intraday_data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=intraday_data.index,
                        y=intraday_data['Close'],
                        mode='lines+markers',
                        line=dict(color='skyblue'),
                        marker=dict(size=4),
                        name='Live Price'
                    ))
                    fig.update_layout(
                        xaxis_title='Time',
                        yaxis_title='Price (₹)',
                        title='Intraday Price Movement',
                        template='plotly_dark',
                        xaxis=dict(
                            showgrid=True,
                            tickformat="%H:%M",
                            nticks=10
                        ),
                        yaxis=dict(
                            showgrid=True,
                            tickformat=".2f",
                            automargin=True
                        ),
                        height=500
                    )
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(5)

    except Exception as e:
        st.warning("Live price not available for this stock or source.")

except Exception as e:
    st.error(f"Something went wrong: {e}")
