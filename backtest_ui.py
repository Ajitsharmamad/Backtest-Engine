import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import talib
from datetime import datetime

# Data loading
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

# Chart conversions
def to_renko(df, brick_size):
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

def to_heikin_ashi(df):
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = (df['open'].shift(1) + df['close'].shift(1)) / 2
    ha_open = ha_open.fillna((df['open'].iloc[0] + df['close'].iloc[0]) / 2)
    ha_high = df[['high', 'open', 'close']].max(axis=1)
    ha_low = df[['low', 'open', 'close']].min(axis=1)
    return pd.DataFrame({'time': df['time'], 'open': ha_open, 'high': ha_high, 'low': ha_low, 'close': ha_close})

# Indicator calculations
def calculate_indicators(df, macd_fast=12, macd_slow=26, macd_signal=9, rsi_period=14, vortex_period=14, 
                       ema_period=20, bb_period=20, bb_std=2, keltner_period=20, keltner_mult=2, 
                       stoch_k=14, stoch_d=3, stoch_smooth=3):
    df['macd_line'], df['signal_line'], df['hist'] = talib.MACD(df['close'].values, fastperiod=macd_fast, 
                                                               slowperiod=macd_slow, signalperiod=macd_signal)
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=rsi_period)
    df['vi_plus'], df['vi_minus'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, 
                                                 timeperiod=vortex_period), talib.MINUS_DI(df['high'].values, 
                                                                                          df['low'].values, 
                                                                                          df['close'].values, 
                                                                                          timeperiod=vortex_period)
    df['ema'] = talib.EMA(df['close'].values, timeperiod=ema_period)
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std * df['bb_std'])
    df['bb_lower'] = df['bb_mid'] - (bb_std * df['bb_std'])
    df['keltner_mid'] = df['close'].rolling(window=keltner_period).mean()
    df['keltner_atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=keltner_period)
    df['keltner_upper'] = df['keltner_mid'] + (keltner_mult * df['keltner_atr'])
    df['keltner_lower'] = df['keltner_mid'] - (keltner_mult * df['keltner_atr'])
    df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values, 
                                              fastk_period=stoch_k, slowk_period=stoch_smooth, 
                                              slowd_period=stoch_d)
    return df

# Strategy builder (simplified Streak clone)
def build_strategy(df, conditions):
    signals = pd.DataFrame(index=df.index, columns=['signal'], data=0)
    for i in range(1, len(df)):
        signal = 0
        for cond in conditions:
            if cond['type'] == 'entry':
                if eval(cond['rule']):
                    signal = 1
            elif cond['type'] == 'exit':
                if eval(cond['rule']):
                    signal = -1
        signals['signal'].iloc[i] = signal
    return signals

# Backtest
def backtest(df, signals, initial_equity=10000, risk_per_trade=0.1, sl_percent=0.02, tp_percent=0.04):
    equity = initial_equity
    position = 0
    entry_price = 0
    trade_log = []
    equity_history = [equity]
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        if signals['signal'].iloc[i] == 1 and position == 0:
            position = equity * risk_per_trade / current_price
            entry_price = current_price
            sl_price = entry_price * (1 - sl_percent)
            tp_price = entry_price * (1 + tp_percent)
            trade_log.append({'type': 'buy', 'price': entry_price, 'sl': sl_price, 'tp': tp_price, 'time': df['time'].iloc[i]})
        elif signals['signal'].iloc[i] == -1 and position == 0:
            position = -equity * risk_per_trade / current_price
            entry_price = current_price
            sl_price = entry_price * (1 + sl_percent)
            tp_price = entry_price * (1 - tp_percent)
            trade_log.append({'type': 'sell', 'price': entry_price, 'sl': sl_price, 'tp': tp_price, 'time': df['time'].iloc[i]})
        elif position != 0:
            if position > 0 and (current_price <= sl_price or current_price >= tp_price or signals['signal'].iloc[i] == -1):
                profit = position * (current_price - entry_price)
                equity += profit
                trade_log.append({'type': 'exit', 'price': current_price, 'profit': profit, 'time': df['time'].iloc[i]})
                position = 0
            elif position < 0 and (current_price >= sl_price or current_price <= tp_price or signals['signal'].iloc[i] == 1):
                profit = position * (current_price - entry_price)
                equity += profit
                trade_log.append({'type': 'exit', 'price': current_price, 'profit': profit, 'time': df['time'].iloc[i]})
                position = 0
        equity_history.append(equity)
    equity_series = pd.Series(equity_history, index=pd.concat([pd.Series([df['time'].iloc[0]]), df['time']]).reset_index(drop=True))
    return equity, trade_log, equity_series

# Scanner (basic)
def scanner(df, condition):
    return df[eval(condition)]

# UI
st.sidebar.header("UI Settings")
theme = st.sidebar.selectbox("Theme", ["Light", "Dark", "Auto"])
font_size = st.sidebar.selectbox("Font Size", ["Small", "Medium", "Large"])
chart_style = st.sidebar.selectbox("Chart Style", ["Solid", "Dashed", "Filled"])
color_scheme = st.sidebar.selectbox("Color Scheme", ["Default", "High Contrast", "Monochrome"])

if theme == "Dark":
    st.markdown("""
        <style>
        .stApp {background-color: #1e1e1e; color: #ffffff;}
        </style>
        """, unsafe_allow_html=True)
elif theme == "Auto":
    st.markdown("""
        <style>
        .stApp {background-color: #f0f0f0; color: #000000;}
        @media (prefers-color-scheme: dark) {
            .stApp {background-color: #1e1e1e; color: #ffffff;}
        }
        </style>
        """, unsafe_allow_html=True)

if font_size == "Large":
    st.markdown("<style>body {font-size: 18px;}</style>", unsafe_allow_html=True)
elif font_size == "Small":
    st.markdown("<style>body {font-size: 12px;}</style>", unsafe_allow_html=True)

st.title("Natural Gas Backtest Engine")
tabs = st.tabs(["Settings", "Strategy Builder", "Backtest", "Scanner"])

with tabs[0]:  # Settings
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if not uploaded_file:
        st.warning("Please upload a CSV file with 'time', 'open', 'high', 'low', 'close', 'EMA', 'Fisher', 'Trigger' columns.")
        st.stop()
    brick_size = st.text_input("Brick Size", value="0.5")
    try:
        brick_size = float(brick_size)
    except ValueError:
        st.error("Invalid Brick Size. Use a number.")
        st.stop()
    risk = st.text_input("Risk per Trade (%)", value="0.1")
    try:
        risk = float(risk)
    except ValueError:
        st.error("Invalid Risk per Trade. Use a number.")
        st.stop()
    sl = st.text_input("Stop-Loss (%)", value="0.02")
    try:
        sl = float(sl)
    except ValueError:
        st.error("Invalid Stop-Loss. Use a number.")
        st.stop()
    tp = st.text_input("Take-Profit (%)", value="0.04")
    try:
        tp = float(tp)
    except ValueError:
        st.error("Invalid Take-Profit. Use a number.")
        st.stop()
    macd_fast = st.text_input("MACD Fast Length", value="12")
    try:
        macd_fast = int(macd_fast)
    except ValueError:
        st.error("Invalid MACD Fast Length. Use an integer.")
        st.stop()
    macd_slow = st.text_input("MACD Slow Length", value="26")
    try:
        macd_slow = int(macd_slow)
    except ValueError:
        st.error("Invalid MACD Slow Length. Use an integer.")
        st.stop()
    macd_signal = st.text_input("MACD Signal Length", value="9")
    try:
        macd_signal = int(macd_signal)
    except ValueError:
        st.error("Invalid MACD Signal Length. Use an integer.")
        st.stop()
    rsi_period = st.text_input("RSI Period", value="14")
    try:
        rsi_period = int(rsi_period)
    except ValueError:
        st.error("Invalid RSI Period. Use an integer.")
        st.stop()
    vortex_period = st.text_input("Vortex Period", value="14")
    try:
        vortex_period = int(vortex_period)
    except ValueError:
        st.error("Invalid Vortex Period. Use an integer.")
        st.stop()
    ema_period = st.text_input("EMA Period", value="20")
    try:
        ema_period = int(ema_period)
    except ValueError:
        st.error("Invalid EMA Period. Use an integer.")
        st.stop()
    bb_period = st.text_input("Bollinger Bands Period", value="20")
    try:
        bb_period = int(bb_period)
    except ValueError:
        st.error("Invalid Bollinger Bands Period. Use an integer.")
        st.stop()
    bb_std = st.text_input("Bollinger Bands Std Dev", value="2")
    try:
        bb_std = float(bb_std)
    except ValueError:
        st.error("Invalid Bollinger Bands Std Dev. Use a number.")
        st.stop()
    keltner_period = st.text_input("Keltner Channels Period", value="20")
    try:
        keltner_period = int(keltner_period)
    except ValueError:
        st.error("Invalid Keltner Channels Period. Use an integer.")
        st.stop()
    keltner_mult = st.text_input("Keltner Channels Multiplier", value="2")
    try:
        keltner_mult = float(keltner_mult)
    except ValueError:
        st.error("Invalid Keltner Channels Multiplier. Use a number.")
        st.stop()
    stoch_k = st.text_input("Stochastic %K Period", value="14")
    try:
        stoch_k = int(stoch_k)
    except ValueError:
        st.error("Invalid Stochastic %K Period. Use an integer.")
        st.stop()
    stoch_d = st.text_input("Stochastic %D Period", value="3")
    try:
        stoch_d = int(stoch_d)
    except ValueError:
        st.error("Invalid Stochastic %D Period. Use an integer.")
        st.stop()
    stoch_smooth = st.text_input("Stochastic Smooth", value="3")
    try:
        stoch_smooth = int(stoch_smooth)
    except ValueError:
        st.error("Invalid Stochastic Smooth. Use an integer.")
        st.stop()

    df = load_and_validate_data(uploaded_file.getvalue())
    if df is not None:
        chart_type = st.selectbox("Chart Type", ["Candle", "Heikin Ashi", "Renko"])
        if chart_type == "Renko":
            chart_df = to_renko(df, brick_size)
        elif chart_type == "Heikin Ashi":
            chart_df = to_heikin_ashi(df)
        else:
            chart_df = df[['time', 'open', 'high', 'low', 'close']]
        chart_df = calculate_indicators(chart_df, macd_fast, macd_slow, macd_signal, rsi_period, vortex_period, 
                                       ema_period, bb_period, bb_std, keltner_period, keltner_mult, 
                                       stoch_k, stoch_d, stoch_smooth)

with tabs[1]:  # Strategy Builder
    st.subheader("Build Your Strategy")
    conditions = []
    entry_conditions = st.text_area("Entry Conditions (e.g., 'df['macd_line'] > df['signal_line']')", 
                                   value="df['Fisher'] > df['Trigger']")
    exit_conditions = st.text_area("Exit Conditions (e.g., 'df['rsi'] > 70')", 
                                  value="df['Fisher'] < df['Trigger']")
    if st.button("Add Strategy"):
        conditions.append({'type': 'entry', 'rule': entry_conditions})
        conditions.append({'type': 'exit', 'rule': exit_conditions})
        st.session_state.conditions = conditions
    if 'conditions' in st.session_state:
        st.write("Current Strategy:", st.session_state.conditions)
    if st.button("Save Strategy"):
        strategy_text = str(st.session_state.conditions)
        st.download_button("Download Strategy", strategy_text, file_name="strategy.txt")
    uploaded_strategy = st.file_uploader("Upload Strategy", type="txt")
    if uploaded_strategy:
        st.session_state.conditions = eval(uploaded_strategy.read().decode())

with tabs[2]:  # Backtest
    if 'conditions' in st.session_state and df is not None:
        signals = build_strategy(chart_df, st.session_state.conditions)
        equity, trade_log, equity_series = backtest(chart_df, signals, initial_equity=10000, risk_per_trade=risk, sl_percent=sl, tp_percent=tp)
        if equity is not None:
            st.write(f"Final Equity: ${equity:.2f}")
            st.write("Max Drawdown: TBD")  # Placeholder for calc
            st.write("Win Rate: TBD")     # Placeholder for calc
            st.write("Trade Log:")
            trade_df = pd.DataFrame(trade_log)
            trade_df['time'] = pd.to_datetime(trade_df['time'])
            st.table(trade_df)
            if chart_style == "Dashed":
                st.line_chart(equity_series, use_container_width=True, line_style="dashed")
            elif chart_style == "Filled":
                st.area_chart(equity_series, use_container_width=True)
            else:
                st.line_chart(equity_series, use_container_width=True)
            if st.button("Deploy Strategy"):
                st.success("Strategy deployed! Check logs for alerts.")

with tabs[3]:  # Scanner
    scan_condition = st.text_input("Scan Condition (e.g., 'df['rsi'] > 70')", value="df['rsi'] > 70")
    if st.button("Scan"):
        scanned_df = scanner(chart_df, scan_condition)
        st.write("Scanned Results:", scanned_df)

if color_scheme == "High Contrast":
    st.markdown("<style>.stChart {color: #ff0000;}</style>", unsafe_allow_html=True)
elif color_scheme == "Monochrome":
    st.markdown("<style>.stChart {color: #000000;}</style>", unsafe_allow_html=True)
