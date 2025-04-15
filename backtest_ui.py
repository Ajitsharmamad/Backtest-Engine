import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO  # Correct import for StringIO

# Data loading with better error handling
def load_and_validate_data(file_content):
    try:
        string_io = StringIO(file_content.decode('utf-8'))
        try:
            df = pd.read_csv(string_io)
        except UnicodeDecodeError:
            string_io = StringIO(file_content.decode('latin-1'))
            df = pd.read_csv(string_io)
        df['time'] = pd.to_datetime(df['time'])
        required_cols = ['time', 'open', 'high', 'low', 'close', 'EMA', 'Fisher', 'Trigger']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required columns")
        if df[required_cols].isna().any().any():
            raise ValueError("Data contains NaN values")
        return df
    except Exception as e:
        st.error(f"Data loading error: {e}")
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
        st.error(f"Renko conversion error: {e}")
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
        equity_series = pd.Series(equity_history, index=pd.concat([pd.Series([renko_df['time'].iloc[0]]), renko_df['time']]).reset_index(drop=True))
        return equity, trade_log, equity_series
    except Exception as e:
        st.error(f"Backtest error: {e}")
        return None, None, None

# Strategy (simple Fisher for now)
def fisher_trigger_strategy(renko_df):
    try:
        signals = pd.DataFrame(index=renko_df.index, columns=['signal'], data=0)
        for i in range(1, len(renko_df)):
            if renko_df['Fisher'].iloc[i] > renko_df['Trigger'].iloc[i] and renko_df['Fisher'].iloc[i-1] <= renko_df['Trigger'].iloc[i-1]:
                signals['signal'].iloc[i] = 1
            elif renko_df['Fisher'].iloc[i] < renko_df['Trigger'].iloc[i] and renko_df['Fisher'].iloc[i-1] >= renko_df['Trigger'].iloc[i-1]:
                signals['signal'].iloc[i] = -1
        return signals
    except Exception as e:
        st.error(f"Strategy error: {e}")
        return None

# Streamlit UI
st.title("Natural Gas Backtest Engine")
st.sidebar.header("Settings")

# Manual input fields replacing sliders
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
if not uploaded_file:
    st.warning("Please upload a CSV file with 'time', 'open', 'high', 'low', 'close', 'EMA', 'Fisher', 'Trigger' columns to proceed.")
    st.stop()
brick_size = st.sidebar.text_input("Brick Size", value="0.5")
try:
    brick_size = float(brick_size)
except ValueError:
    st.sidebar.error("Invalid Brick Size. Use a number (e.g., 0.5)")
    st.stop()
risk = st.sidebar.text_input("Risk per Trade (%)", value="0.1")
try:
    risk = float(risk)
except ValueError:
    st.sidebar.error("Invalid Risk per Trade. Use a number (e.g., 0.1)")
    st.stop()
sl = st.sidebar.text_input("Stop-Loss (%)", value="0.02")
try:
    sl = float(sl)
except ValueError:
    st.sidebar.error("Invalid Stop-Loss. Use a number (e.g., 0.02)")
    st.stop()
tp = st.sidebar.text_input("Take-Profit (%)", value="0.04")
try:
    tp = float(tp)
except ValueError:
    st.sidebar.error("Invalid Take-Profit. Use a number (e.g., 0.04)")
    st.stop()

if uploaded_file is not None:
    df = load_and_validate_data(uploaded_file.getvalue())
    if df is not None:
        renko_df = to_renko(df, brick_size)
        if renko_df is not None:
            signals = fisher_trigger_strategy(renko_df)
            if signals is not None:
                equity, trade_log, equity_series = backtest(renko_df, signals, initial_equity=10000, risk_per_trade=risk, sl_percent=sl, tp_percent=tp)
                if equity is not None:
                    st.write(f"Final Equity: ${equity:.2f}")
                    st.write("Trade Log:")
                    trade_df = pd.DataFrame(trade_log)
                    trade_df['time'] = pd.to_datetime(trade_df['time'])
                    st.table(trade_df)
                    st.line_chart(equity_series)

if st.button("Run Backtest"):
    st.write("Backtest running... (Refresh after upload to see results)")
