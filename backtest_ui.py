import pandas as pd
import numpy as np
import streamlit as st

# Data loading
def load_and_validate_data(file_content):
    try:
        df = pd.read_csv(pd.StringIO(file_content))
        df['time'] = pd.to_datetime(df['time'])
        required_cols = ['time', 'open', 'high', 'low', 'close', 'EMA', 'Fisher', 'Trigger']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required columns")
        if df[required_cols].isna().any().any():
            raise ValueError("Data contains NaN values")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Renko conversion
def to_renko(df, brick_size=0.5):
    try:
        df = df.sort_values('time')
        renko = []
        price = df['close'].iloc[0]
        direction = 0
        for index, row in df.iterrows():
            current_price = row['close']
            while abs(current_price - price) >= brick_size:
                move = brick_size if current_price > price else -brick_size
                new_price = price + move
                if (move > 0 and direction <= 0) or (move < 0 and direction >= 0):
                    renko.append({'time': row['time'], 'close': new_price, 'Fisher': row['Fisher'], 'Trigger': row['Trigger']})
                    direction = 1 if move > 0 else -1
                price = new_price
            if abs(current_price - price) < brick_size:
                renko.append({'time': row['time'], 'close': price, 'Fisher': row['Fisher'], 'Trigger': row['Trigger']})
        return pd.DataFrame(renko)
    except Exception as e:
        st.error(f"Error converting to Renko: {e}")
        return None

# Strategy
def fisher_trigger_strategy(renko_df):
    try:
        signals = pd.DataFrame(index=renko_df.index, columns=['signal'], data=0)
        for i in range(1, len(renko_df)):
            if (renko_df['Fisher'].iloc[i] > renko_df['Trigger'].iloc[i] and 
                renko_df['Fisher'].iloc[i-1] <= renko_df['Trigger'].iloc[i-1]):
                signals['signal'].iloc[i] = 1
            elif (renko_df['Fisher'].iloc[i] < renko_df['Trigger'].iloc[i] and 
                  renko_df['Fisher'].iloc[i-1] >= renko_df['Trigger'].iloc[i-1]):
                signals['signal'].iloc[i] = -1
        return signals
    except Exception as e:
        st.error(f"Error in strategy: {e}")
        return None

# Backtest with SL/TP
def backtest(renko_df, signals, initial_equity=10000, risk_per_trade=0.1, sl_percent=0.02, tp_percent=0.04):
    try:
        equity = initial_equity
        position = 0
        entry_price = 0
        trade_log = []
        equity_history = [initial_equity]
        for i in range(len(renko_df)):
            current_price = renko_df['close'].iloc[i]
            if signals['signal'].iloc[i] == 1 and position == 0:
                position = equity * risk_per_trade / current_price
                entry_price = current_price
                sl_price = entry_price * (1 - sl_percent)
                tp_price = entry_price * (1 + tp_percent)
                trade_log.append({'type': 'buy', 'price': entry_price, 'sl': sl_price, 'tp': tp_price, 'time': renko_df['time'].iloc[i]})
            elif signals['signal'].iloc[i] == -1 and position == 0:
                position = -equity * risk_per_trade / current_price
                entry_price = current_price
                sl_price = entry_price * (1 + sl_percent)
                tp_price = entry_price * (1 - tp_percent)
                trade_log.append({'type': 'sell', 'price': entry_price, 'sl': sl_price, 'tp': tp_price, 'time': renko_df['time'].iloc[i]})
            elif position != 0:
                if position > 0 and (current_price <= sl_price or current_price >= tp_price or signals['signal'].iloc[i] == -1):
                    profit = position * (current_price - entry_price)
                    equity += profit
                    trade_log.append({'type': 'exit', 'price': current_price, 'profit': profit, 'time': renko_df['time'].iloc[i]})
                    position = 0
                elif position < 0 and (current_price >= sl_price or current_price <= tp_price or signals['signal'].iloc[i] == 1):
                    profit = position * (current_price - entry_price)
                    equity += profit
                    trade_log.append({'type': 'exit', 'price': current_price, 'profit': profit, 'time': renko_df['time'].iloc[i]})
                    position = 0
            equity_history.append(equity)
        return equity, trade_log, equity_history
    except Exception as e:
        st.error(f"Error in backtest: {e}")
        return None, None, None

# Streamlit UI
st.title("Natural Gas Backtest Engine")
st.sidebar.header("Settings")

# Input fields
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
brick_size = st.sidebar.slider("Brick Size", 0.1, 1.0, 0.5, 0.1)
risk = st.sidebar.slider("Risk per Trade (%)", 0.1, 1.0, 0.1, 0.1)
sl = st.sidebar.slider("Stop-Loss (%)", 0.01, 0.05, 0.02, 0.01)
tp = st.sidebar.slider("Take-Profit (%)", 0.02, 0.10, 0.04, 0.01)

if uploaded_file is not None:
    df = load_and_validate_data(uploaded_file.getvalue().decode())
    if df is not None:
        renko_df = to_renko(df, brick_size)
        if renko_df is not None:
            signals = fisher_trigger_strategy(renko_df)
            if signals is not None:
                equity, trade_log, equity_history = backtest(renko_df, signals, initial_equity=10000, risk_per_trade=risk, sl_percent=sl, tp_percent=tp)
                if equity is not None:
                    st.write(f"Final Equity: ${equity:.2f}")
                    st.write("Trade Log:")
                    trade_df = pd.DataFrame(trade_log)
                    trade_df['time'] = pd.to_datetime(trade_df['time'])
                    st.table(trade_df)
                    # Equity curve with time index
                    equity_series = pd.Series(equity_history, index=renko_df['time'])
                    st.line_chart(equity_series)

if st.button("Run Backtest"):
    st.write("Backtest running... (Refresh after upload to see results)")
