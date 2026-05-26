import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# Configure page
st.set_page_config(page_title="LegendStock - LSTM Prediction", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
    }
    .prediction-price {
        font-size: 48px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("Stock Market Trend Prediction Using LSTM")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose Analysis",
    ["Single Stock Prediction", "Technical Analysis", "Compare Stocks", "Portfolio Tracker"])

# Common parameters
stock = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
start = st.sidebar.date_input("Start Date", datetime(2015, 1, 1))
end = st.sidebar.date_input("End Date", datetime(2026, 1, 1))

# Validate dates
if start >= end:
    st.error("Start date must be before end date")
    st.stop()

@st.cache_data
def load_data(ticker, start_date, end_date):
    """Load stock data with caching for performance"""
    try:
        data = yf.download(ticker, start_date, end_date)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    df = df.copy()

    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

    # Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)

    return df

def get_trading_signal(df):
    """Generate trading signals based on technical indicators"""
    latest = df.iloc[-1]
    signals = []

    # SMA signals
    if latest['Close'] > latest['SMA_50']:
        signals.append(("Price above SMA-50 (Bullish)", "bullish"))
    else:
        signals.append(("Price below SMA-50 (Bearish)", "bearish"))

    # MACD signal
    if latest['MACD'] > latest['MACD_Signal']:
        signals.append(("MACD crossing above signal (Bullish)", "bullish"))
    else:
        signals.append(("MACD below signal (Bearish)", "bearish"))

    # RSI signal
    if latest['RSI'] > 70:
        signals.append(("RSI indicates overbought (Caution)", "bearish"))
    elif latest['RSI'] < 30:
        signals.append(("RSI indicates oversold (Opportunity)", "bullish"))
    else:
        signals.append(("RSI in neutral zone", "neutral"))

    # Bollinger Bands
    if latest['Close'] > latest['BB_Upper']:
        signals.append(("Price above upper BB (Overbought)", "bearish"))
    elif latest['Close'] < latest['BB_Lower']:
        signals.append(("Price below lower BB (Oversold)", "bullish"))
    else:
        signals.append(("Price within Bollinger Bands", "neutral"))

    return signals

data = load_data(stock, start, end)

if data is None or len(data) == 0:
    st.error("Invalid stock symbol or no data available")
    st.stop()

if page == "Single Stock Prediction":
    st.subheader("Stock Data Overview")
    st.write(data.tail())

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}",
                 f"{((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100:.2f}%")
    with col2:
        st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
    with col3:
        st.metric("52-Week High", f"${data['High'].max():.2f}")
    with col4:
        st.metric("52-Week Low", f"${data['Low'].min():.2f}")

    st.divider()

    # LSTM Model Parameters
    st.subheader("LSTM Model Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        lookback = st.slider("Lookback Days", 30, 120, 60)
    with col2:
        epochs = st.slider("Training Epochs", 1, 50, 20)
    with col3:
        train_split = st.slider("Train/Test Split (%)", 60, 90, 80)

    close_price = data[['Close']]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(close_price)

    train_size = int(len(scaled) * train_split / 100)

    train = scaled[:train_size]

    X_train=[]
    Y_train=[]

    for i in range(lookback, len(train)):
        X_train.append(train[i-lookback:i,0])
        Y_train.append(train[i,0])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X_train = np.reshape(
        X_train,
        (X_train.shape[0],
         X_train.shape[1],
         1)
    )

    # Build model with dropout for better generalization
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner(f"Training LSTM model with {epochs} epochs..."):
        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            validation_split=0.1
        )

    # Plot training loss
    fig_loss = plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(fig_loss)

    # Prepare test data
    inputs = scaled[train_size-lookback:]
    X_test = []

    for i in range(lookback, len(inputs)):
        X_test.append(inputs[i-lookback:i,0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Make predictions
    pred = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(pred)

    actual = close_price[train_size:]

    # Calculate prediction metrics
    mse = np.mean((actual.values - pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual.values - pred))
    mape = np.mean(np.abs((actual.values - pred) / actual.values)) * 100

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RMSE", f"${rmse:.2f}")
    with col2:
        st.metric("MAE", f"${mae:.2f}")
    with col3:
        st.metric("MAPE", f"{mape:.2f}%")

    # Plot predictions
    st.subheader("Actual vs Predicted Prices")
    fig = plt.figure(figsize=(14, 6))
    plt.plot(actual.values, label="Actual Price", linewidth=2)
    plt.plot(pred, label="Predicted Price", linewidth=2, alpha=0.8)
    plt.fill_between(range(len(pred)), actual.values.flatten(), pred.flatten(),
                     alpha=0.3, color='gray')
    plt.title(f'{stock} - Actual vs Predicted Stock Prices')
    plt.xlabel('Trading Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Multi-day future predictions
    st.subheader("Future Price Predictions")

    days_ahead = st.slider("Predict Days Ahead", 1, 30, 7)

    future_predictions = []
    last_sequence = scaled[-lookback:].copy()

    for day in range(days_ahead):
        next_input = last_sequence.reshape(1, lookback, 1)
        next_pred = model.predict(next_input, verbose=0)
        future_predictions.append(scaler.inverse_transform(next_pred)[0][0])
        last_sequence = np.append(last_sequence[1:], next_pred)

    # Display predictions
    prediction_dates = [data.index[-1] + timedelta(days=i+1) for i in range(days_ahead) if data.index[-1].dayofweek + i + 1 <= 5]

    fig_future = plt.figure(figsize=(12, 5))
    plt.plot(range(len(actual)), actual.values, label='Historical', linewidth=2)
    plt.plot(range(len(actual), len(actual) + days_ahead),
             future_predictions, label='Future Predictions',
             linestyle='--', marker='o', markersize=6)
    plt.axvline(x=len(actual), color='red', linestyle=':', label='Prediction Start')
    plt.title(f'{stock} - {days_ahead}-Day Price Forecast')
    plt.xlabel('Trading Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_future)

    # Display prediction summary
    st.subheader("Prediction Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Next Day", f"${future_predictions[0]:.2f}")
    with col2:
        if len(future_predictions) >= 7:
            st.metric("Next Week (Avg)", f"${np.mean(future_predictions[:7]):.2f}")
    with col3:
        st.metric(f"Day {days_ahead}", f"${future_predictions[-1]:.2f}")

elif page == "Technical Analysis":
    st.subheader(f"Technical Analysis for {stock}")

    # Calculate indicators
    data = calculate_technical_indicators(data)

    # Plot candlestick chart
    from matplotlib.dates import DateFormatter
    fig_candle = plt.figure(figsize=(14, 8))

    # Create OHLC plot
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(data.index, data['Close'], label='Close Price', linewidth=1.5)
    plt.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
    plt.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7)
    plt.plot(data.index, data['SMA_200'], label='SMA 200', alpha=0.7)
    plt.fill_between(data.index, data['BB_Upper'], data['BB_Lower'],
                     alpha=0.2, color='gray', label='Bollinger Bands')
    plt.title(f'{stock} - Price and Moving Averages')
    plt.ylabel('Price ($)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # MACD plot
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(data.index, data['MACD'], label='MACD', linewidth=1.5)
    plt.plot(data.index, data['MACD_Signal'], label='Signal Line', alpha=0.7)
    plt.bar(data.index, data['MACD_Histogram'],
            color=['g' if x > 0 else 'r' for x in data['MACD_Histogram']],
            alpha=0.3, label='Histogram')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('MACD Indicator')
    plt.ylabel('MACD')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # RSI plot
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(data.index, data['RSI'], label='RSI', linewidth=1.5)
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    plt.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
    plt.title('RSI Indicator')
    plt.ylabel('RSI')
    plt.xlabel('Date')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_candle)

    # Trading signals
    st.subheader("Trading Signals")
    signals = get_trading_signal(data)

    for signal, signal_type in signals:
        if signal_type == "bullish":
            st.success(signal)
        elif signal_type == "bearish":
            st.error(signal)
        else:
            st.info(signal)

    # Display latest indicators
    st.subheader("Latest Indicator Values")
    latest = data.iloc[-1]
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("RSI", f"{latest['RSI']:.2f}")
    with col2:
        st.metric("MACD", f"{latest['MACD']:.4f}")
    with col3:
        st.metric("Volatility", f"{latest['Volatility_20']:.2%}")
    with col4:
        st.metric("SMA 20", f"${latest['SMA_20']:.2f}")
    with col5:
        st.metric("SMA 50", f"${latest['SMA_50']:.2f}")
    with col6:
        st.metric("BB Upper", f"${latest['BB_Upper']:.2f}")

elif page == "Compare Stocks":
    st.subheader("Compare Multiple Stocks")

    # Input for multiple stocks
    stocks_input = st.text_input(
        "Enter stock symbols (comma-separated)",
        "AAPL, GOOGL, MSFT"
    )
    stocks = [s.strip().upper() for s in stocks_input.split(",")]

    if st.button("Compare Stocks"):
        compare_data = {}
        for s in stocks:
            stock_data = load_data(s, start, end)
            if stock_data is not None and len(stock_data) > 0:
                compare_data[s] = stock_data['Close']

        if compare_data:
            df_compare = pd.DataFrame(compare_data)

            # Normalize prices for comparison
            df_normalized = df_compare / df_compare.iloc[0] * 100

            # Plot comparison
            fig_compare = plt.figure(figsize=(14, 6))
            for col in df_normalized.columns:
                plt.plot(df_normalized.index, df_normalized[col],
                        label=col, linewidth=2)
            plt.title('Stock Performance Comparison (Normalized)')
            plt.xlabel('Date')
            plt.ylabel('Normalized Price (%)')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_compare)

            # Display statistics
            st.subheader("Performance Statistics")
            stats_df = pd.DataFrame({
                'Stock': stocks,
                'Current Price': [df_compare[s].iloc[-1] if s in df_compare.columns else np.nan for s in stocks],
                'Start Price': [df_compare[s].iloc[0] if s in df_compare.columns else np.nan for s in stocks],
                'Return (%)': [((df_compare[s].iloc[-1] / df_compare[s].iloc[0] - 1) * 100) if s in df_compare.columns else np.nan for s in stocks],
                'Volatility': [df_compare[s].pct_change().std() * np.sqrt(252) if s in df_compare.columns else np.nan for s in stocks]
            })
            st.dataframe(stats_df.style.format({
                'Current Price': '${:.2f}',
                'Start Price': '${:.2f}',
                'Return (%)': '{:.2f}%',
                'Volatility': '{:.2%}'
            }))

elif page == "Portfolio Tracker":
    st.subheader("Portfolio Performance Tracker")

    # Portfolio input
    st.write("Enter your portfolio holdings:")
    num_holdings = st.number_input("Number of holdings", min_value=1, max_value=20, value=3)

    portfolio_data = []
    for i in range(num_holdings):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input(f"Stock {i+1}", key=f"ticker_{i}").upper()
        with col2:
            shares = st.number_input(f"Shares", min_value=0.0, key=f"shares_{i}")
        with col3:
            buy_price = st.number_input(f"Buy Price ($)", min_value=0.0, key=f"price_{i}")

        if ticker and shares > 0 and buy_price > 0:
            portfolio_data.append({
                'ticker': ticker,
                'shares': shares,
                'buy_price': buy_price
            })

    if st.button("Track Portfolio") and portfolio_data:
        portfolio_value = []
        total_investment = 0
        total_current_value = 0

        for holding in portfolio_data:
            ticker = holding['ticker']
            shares = holding['shares']
            buy_price = holding['buy_price']

            stock_data = load_data(ticker, start, end)
            if stock_data is not None and len(stock_data) > 0:
                current_price = stock_data['Close'].iloc[-1]
                current_value = current_price * shares
                invested = buy_price * shares
                pnl = current_value - invested
                pnl_pct = (pnl / invested) * 100

                total_investment += invested
                total_current_value += current_value

                portfolio_value.append({
                    'Ticker': ticker,
                    'Shares': shares,
                    'Buy Price': f'${buy_price:.2f}',
                    'Current Price': f'${current_price:.2f}',
                    'Invested': f'${invested:,.2f}',
                    'Current Value': f'${current_value:,.2f}',
                    'P&L ($)': f'${pnl:,.2f}',
                    'P&L (%)': f'{pnl_pct:.2f}%',
                    'Current Price Num': current_price,
                    'P&L Num': pnl
                })

        if portfolio_value:
            portfolio_df = pd.DataFrame(portfolio_value)

            # Display portfolio table
            st.subheader("Portfolio Holdings")
            display_df = portfolio_df.drop(['Current Price Num', 'P&L Num'], axis=1)
            st.dataframe(display_df, use_container_width=True)

            # Portfolio summary
            st.subheader("Portfolio Summary")
            total_pnl = total_current_value - total_investment
            total_pnl_pct = (total_pnl / total_investment) * 100 if total_investment > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Investment", f"${total_investment:,.2f}")
            with col2:
                st.metric("Current Value", f"${total_current_value:,.2f}")
            with col3:
                st.metric("Total P&L", f"${total_pnl:,.2f}")
            with col4:
                st.metric("Total Return", f"{total_pnl_pct:.2f}%")

            # Portfolio allocation chart
            fig_pie = plt.figure(figsize=(8, 6))
            sizes = portfolio_df['Current Price Num'] * [h['shares'] for h in portfolio_data]
            labels = [h['ticker'] for h in portfolio_data]
            colors = plt.cm.Set3(np.linspace(0, 1, len(portfolio_data)))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            plt.title('Portfolio Allocation by Value')
            st.pyplot(fig_pie)
