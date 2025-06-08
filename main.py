from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import requests
import pandas as pd
import ta
import numpy as np
import os
import time
from datetime import timedelta
from datetime import datetime, timezone

# === CONFIGURATION ===
TIMEFRAME = os.getenv("SCAN_TIMEFRAME") or '1h'  # or '1h', '1d'
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL") or "YOUR_DISCORD_WEBHOOK_URL"

# == CONFIGURATION FOR DIP BUYING SIGNAL == 
LOOKBACK = 26
RSI2_THRESHOLD = 4
RSI14_LOW = 25
MFI_LOW = 26

# === TELEGRAM ALERT ===
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg})

# === DISCORD ALERT
def send_discord(msg):
    if DISCORD_WEBHOOK_URL and DISCORD_WEBHOOK_URL.startswith("https://"):
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})


#DICORD SEND IMAGE
def send_chart_to_discord(image_path, webhook_url=DISCORD_WEBHOOK_URL, message="📊 Signal Chart"):
    try:
        with open(image_path, 'rb') as f:
            file_data = {'file': (os.path.basename(image_path), f, 'image/png')}
            payload = {
                'content': message,
                'username': 'SignalBot'
            }

            response = requests.post(webhook_url, data=payload, files=file_data)

            if response.status_code in [200, 204]:
                print("✅ Image sent to Discord successfully.")
            else:
                print(f"❌ Discord upload failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Exception while sending image to Discord: {e}")
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
            print("🗑️ Deleted temporary file.")


# === FETCH BINANCE TOP 100 SYMBOLS ===
def get_top_100_usdt_symbols():
    url = 'https://api.binance.com/api/v3/ticker/24hr'
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        usdt_pairs = [x['symbol'] for x in data if isinstance(x, dict) and x.get('symbol', '').endswith('USDT') and not x['symbol'].endswith('BUSD')]
        sorted_pairs = sorted(usdt_pairs, key=lambda x: float(next(item for item in data if item.get('symbol') == x)['quoteVolume']), reverse=True)
        return sorted_pairs[:68]
    except Exception as e:
        print(f"🚨 Error fetching top 100 symbols: {e}")
        return []


# === FETCH KLINES ===
def fetch_klines(symbol, interval= TIMEFRAME, limit=100):
    url = f"https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    res = requests.get(url, params=params).json()
    df = pd.DataFrame(res, columns=['time','open','high','low','close','volume','ct','qv','ntr','tbb','tbq','ignore'])
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df

#SUPERTREND CALCULATION 

def calculate_supertrend(df, period=10, multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)

    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)

    supertrend = [np.nan] * len(df)
    direction = [True] * len(df)  # True = uptrend, False = downtrend

    for i in range(period, len(df)):
        if df['close'][i] > final_upperband[i - 1]:
            direction[i] = True
        elif df['close'][i] < final_lowerband[i - 1]:
            direction[i] = False
        else:
            direction[i] = direction[i - 1]
            if direction[i] and final_lowerband[i] < supertrend[i - 1]:
                final_lowerband[i] = supertrend[i - 1]
            if not direction[i] and final_upperband[i] > supertrend[i - 1]:
                final_upperband[i] = supertrend[i - 1]

        supertrend[i] = final_lowerband[i] if direction[i] else final_upperband[i]

    supertrend_series = pd.Series(supertrend, index=df.index)
    direction_series = pd.Series(direction, index=df.index)
    # ✅ Tách supertrend thành 2 đường để vẽ 2 màu
    supertrend_up = supertrend_series.where(direction_series == True)
    supertrend_down = supertrend_series.where(direction_series == False)

    return supertrend_series, supertrend_up, supertrend_down
# === Heikin Ashi MA Calculation ===

def apply_custom_ma(series, ma_type='EMA', length=9, alma_offset=0.85, alma_sigma=6):
    if ma_type == 'EMA':
        return ta.trend.ema_indicator(series, window=length)
    elif ma_type == 'SMA':
        return ta.trend.sma_indicator(series, window=length)
    elif ma_type == 'WMA':
        return ta.trend.wma_indicator(series, window=length)
    elif ma_type == 'VWMA':
        return (series * series).rolling(length).sum() / series.rolling(length).sum()
    elif ma_type == 'ZLEMA':
        lag = int((length - 1) / 2)
        zlema_input = series + (series - series.shift(lag))
        return ta.trend.ema_indicator(zlema_input, window=length)
    elif ma_type == 'HMA':
        half = int(length / 2)
        sqrt = int(np.sqrt(length))
        wma1 = ta.trend.wma_indicator(series, window=half)
        wma2 = ta.trend.wma_indicator(series, window=length)
        diff = 2 * wma1 - wma2
        return ta.trend.wma_indicator(diff, window=sqrt)
    elif ma_type == 'ALMA':
        weights = np.exp(-((np.arange(length) - alma_offset * (length - 1)) ** 2) / (2 * alma_sigma ** 2))
        weights /= weights.sum()
        return series.rolling(length).apply(lambda x: np.dot(x, weights), raw=True)
    elif ma_type == 'SWMA':
        return series.rolling(length).mean()
    else:
        return ta.trend.ema_indicator(series, window=length)

def compute_trend(df, ma_type='EMA', ma_period=9, alma_offset=0.85, alma_sigma=6):
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = df['close'].shift(1)
    ha_high = df[['high', 'open', 'close']].max(axis=1)
    ha_low = df[['low', 'open', 'close']].min(axis=1)

    ma_ha_close = apply_custom_ma(ha_close, ma_type, ma_period, alma_offset, alma_sigma)
    ma_ha_open = apply_custom_ma(ha_open, ma_type, ma_period, alma_offset, alma_sigma)
    ma_ha_high = apply_custom_ma(ha_high, ma_type, ma_period, alma_offset, alma_sigma)
    ma_ha_low = apply_custom_ma(ha_low, ma_type, ma_period, alma_offset, alma_sigma)

    trend = 100 * (ma_ha_close - ma_ha_open) / (ma_ha_high - ma_ha_low).replace(0, np.nan)
    return trend

#DETECT FATIGUE TREND
def detect_trend_fatigue(df, ema_length=9, lookback=30, flat_threshold_pct=0.4):
    """
    Adds an 'is_fatigue' column to df that marks True if the trend shows fatigue
    based on mean deviation of EMA.

    Parameters:
        df: pd.DataFrame - must contain a 'close' column.
        ema_length: int - the EMA length to use (default: 9)
        lookback: int - the lookback period to compare deviation
        flat_threshold_pct: float - mean deviation threshold (%) for fatigue
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column.")

    df['ema9'] = ta.trend.ema_indicator(df['close'], window=ema_length)

    # Initialize fatigue column
    df['is_fatigue'] = False

    for i in range(lookback, len(df)):
        ema_check = df.iloc[i - lookback]['ema9']
        ema_window = df.iloc[i - lookback + 1:i + 1]['ema9']
        diff_pct = (ema_window - ema_check).abs() / ema_check * 100
        mean_diff_pct = diff_pct.mean()

        df.at[i, 'is_fatigue'] = mean_diff_pct < flat_threshold_pct

    return df

# === BUY1 SIGNAL LOGIC ===
def is_buy1_signal(df: pd.DataFrame) -> bool:
    if df.shape[0] < 60:
        return False

    # Use only closed bars, drop the latest incomplete one
    df = df.iloc[:-1].copy()
    
    #EMA
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)

    # RSI 
    LOOKBACK = 14  # or adjust to your desired value

    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['average_close'] = df['close'].rolling(window=LOOKBACK).mean()
    df['average_rsi'] = df['rsi'].rolling(window=LOOKBACK).mean()
    df['rsi_bull_divergence'] = (df['close'] <= df['average_close']) & (df['rsi'] > df['average_rsi'])
    df['rsi_bear_divergence'] = (df['close'] >= df['average_close']) & (df['rsi'] < df['average_rsi'])

    df['rsi_superoverbought'] = df['rsi'].rolling(18).max() >= 85

    #Money Flow index
    
    df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
    df['average_mfi'] = df['mfi'].rolling(window=LOOKBACK).mean()
    df['mfi_bull_divergence'] = (df['close'] <= df['average_close']) & (df['mfi'] > df['average_mfi'])
    df['mfi_bear_divergence'] = (df['close'] >= df['average_close']) & (df['mfi'] < df['average_mfi'])

    #Heikin Ashi Trend 
    df['trend'] = compute_trend(df)
    df['trend_prev'] = df['trend'].shift(1)
    df['trend_prev2'] = df['trend'].shift(2)
    
    #ADX
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    
    #Ichimoku
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    tenkan = (high_9 + low_9) / 2
    kijun = (high_26 + low_26) / 2
    df['kumo_a'] = ((tenkan + kijun) / 2).shift(25)
    
    #Supertrend
    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=10)
    basic_upperband = (df['high'] + df['low']) / 2 + 3 * atr
    basic_lowerband = (df['high'] + df['low']) / 2 - 3 * atr
    supertrend_direction = np.where(df['close'] > basic_lowerband, 1, -1)
    df['supertrend_dir'] = pd.Series(supertrend_direction).astype(int)
    df['supertrend_change'] = df['supertrend_dir'].diff()
    last_bearish_switch_index = df[df['supertrend_change'] == 2].index.max()
    bearish_switch_not_recent = (df.index[-1] - last_bearish_switch_index) > 9 if last_bearish_switch_index is not None else True
    #Fatigue trend
    df = detect_trend_fatigue(df)
    #Remove not yet closed bar
    latest = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]

    buy1 = (
        latest['close'] >= latest['ema9'] and
        latest['trend'] > 0 and
        prev1['trend'] > 0 and
        prev2['trend'] <= 0 and
        latest['close'] > latest['kumo_a'] and
        latest['adx'] > 13 and
        latest['rsi'] <= 66 and
        bearish_switch_not_recent and
        not latest['rsi_bear_divergence'] and
        not latest['mfi_bear_divergence'] and
        not latest['rsi_superoverbought'] and
        not latest['is_fatigue']
    )
    return bool(buy1)

# === BUY 2 SIGNAL === #
def is_buy2_signal(df, lookback=26):
    if df.shape[0] < lookback + 1:
        return False

    # Drop current (incomplete) bar
    df = df.iloc[:-1].copy()

    # Calculate RSI(2) and RSI(14)
    df['rsi2'] = ta.momentum.rsi(df['close'], window=2)
    df['rsi14'] = ta.momentum.rsi(df['close'], window=14)

    # Rolling lowest values
    df['lowest_close'] = df['close'].rolling(window=lookback).min()
    df['lowest_rsi14'] = df['rsi14'].rolling(window=lookback).min()

    # Defensive check
    if df[['rsi2', 'rsi14', 'lowest_close', 'lowest_rsi14']].iloc[-1].isnull().any():
        return False
    # Get the last completed candle
    latest = df.iloc[-1]

    buy2 = (
        latest['close'] <= latest['lowest_close'] and
        latest['rsi14'] > latest['lowest_rsi14'] and
        latest['lowest_rsi14'] <= 30 and
        latest['rsi2'] < 20 and
        latest['rsi14'] < 33
    )

    return bool(buy2)

# === BUY 3 SIGNAL === #
def is_buy3_signal(df: pd.DataFrame) -> bool:
    if df.shape[0] < 15:  # Make sure there are enough bars
        return False

    df = df.iloc[:-1].copy()  # Remove current incomplete bar

    df['rsi2'] = ta.momentum.rsi(df['close'], window=2)
    df['rsi14'] = ta.momentum.rsi(df['close'], window=14)
    df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)

    # Defensive check
    if df[['rsi2', 'rsi14', 'mfi']].iloc[-1].isnull().any():
        return False

    latest = df.iloc[-1]
    buy3 = (
        latest['rsi2'] <= 4 and
        latest['rsi14'] < 25 and
        latest['mfi'] < 20
    )
    return bool(buy3)

# === FETCH LATEST BTC/USDT PRICE ===
def fetch_latest_price(symbol='BTCUSDT'):
    try:
        url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            print(f"Binance error: Status {res.status_code}, body: {res.text}")
            return None
        data = res.json()
        price = float(data.get('price', 0))
        return price if price > 0 else None
    except Exception as e:
        print(f"Error fetching BTC price: {e}")
        return None

#SELL SIGNAL FOR BTC & ETH
# Global trade count storage
sell_trade_count = {}

def is_sell_signal(df: pd.DataFrame, trade_count: int) -> bool:
    if df.shape[0] < 60:
        return False

    df = df.copy()

    # === Indicators ===

    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    rsi_oversold_level = 40
    rsi_overbought_level = 60
    df['rsi_oversold'] = df['rsi'] <= rsi_oversold_level
    df['rsi_overbought'] = df['rsi'] >= rsi_overbought_level

    adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=True)

    df['adx'] = adx_indicator.adx()
    df['plus_di'] = adx_indicator.adx_pos()   # ✅ correct usage
    df['minus_di'] = adx_indicator.adx_neg()  # ✅ correct usage
    df['down_adx'] = (df['adx'] >= 20) & (df['minus_di'] <= 15)

    low14 = df['low'].rolling(window=14).min()
    high14 = df['high'].rolling(window=14).max()
    df['k'] = 100 * (df['close'] - low14) / (high14 - low14)
    df['k_smooth'] = df['k'].rolling(window=3).mean()
    df['d'] = df['k_smooth'].rolling(window=3).mean()
    df['stoch_overbought'] = df['d'] >= 80

    # === Latest bar condition ===
    latest = df.iloc[-1]

    # === Base Sell Logic ===
    base_sell_condition = (
        not latest['rsi_oversold'] and
        latest['rsi_overbought'] and
        latest['stoch_overbought'] and
        latest['down_adx']
    )

    return bool(base_sell_condition)

# === TEST FETCH PRICE LOCALLY ===
#print(fetch_latest_price())
#CHART THE BUY SIGNAL

def plot_signal_chart(symbol):
    df = fetch_klines(symbol, interval='1h', limit=100)

    # === Format time ===
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    df['date_num'] = mdates.date2num(df['datetime'])
    
    # === Plot Layout ===
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0)
    ax = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax)

    # === Calculate Indicators ===
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['rsi2'] = ta.momentum.rsi(df['close'], window=2)
    df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
    # === Supertrend ===
    df['supertrend'], df['supertrend_up'], df['supertrend_down'] = calculate_supertrend(df)

    # === Ichimoku ===
    ichimoku = ta.trend.IchimokuIndicator(
        high=df['high'], low=df['low'], window1=9, window2=26, window3=52, visual=False, fillna=False
    )
    df['tenkan_sen'] = ichimoku.ichimoku_conversion_line()
    df['kijun_sen'] = ichimoku.ichimoku_base_line()
    df['senkou_span_a'] = ichimoku.ichimoku_a()
    df['senkou_span_b'] = ichimoku.ichimoku_b()
    df['chikou_span'] = df['close'].shift(-26)  # plotted backward

    # === Shift Senkou A & B forward 26 candles
    cloud_offset = 26
    df['senkou_span_a_shifted'] = df['senkou_span_a'].shift(cloud_offset)
    df['senkou_span_b_shifted'] = df['senkou_span_b'].shift(cloud_offset)


    # === Extend future timestamps for cloud (26 periods forward)
    future_dates = [df['datetime'].iloc[-1] + timedelta(hours=4 * (i + 1)) for i in range(cloud_offset)]
    future_date_nums = mdates.date2num(future_dates)

    # Get last 26 values from original senkou A/B to project forward
    future_span_a = df['senkou_span_a'].iloc[-cloud_offset:].tolist()
    future_span_b = df['senkou_span_b'].iloc[-cloud_offset:].tolist()

    # Pad if < 26 data
    future_span_a += [np.nan] * (cloud_offset - len(future_span_a))
    future_span_b += [np.nan] * (cloud_offset - len(future_span_b))

    future_df = pd.DataFrame({
        'date_num': future_date_nums,
        'senkou_span_a_shifted': future_span_a,
        'senkou_span_b_shifted': future_span_b
    })

    # Combine current and future Ichimoku cloud
    cloud_df = pd.concat([
        df[['date_num', 'senkou_span_a_shifted', 'senkou_span_b_shifted']],
        future_df
    ], ignore_index=True)


    # === Plot Cloud
    ax.fill_between(
        cloud_df['date_num'],
        cloud_df['senkou_span_a_shifted'],
        cloud_df['senkou_span_b_shifted'],
        where=(cloud_df['senkou_span_a_shifted'] >= cloud_df['senkou_span_b_shifted']),
        color='green',
        alpha=0.2
    )
    ax.fill_between(
        cloud_df['date_num'],
        cloud_df['senkou_span_a_shifted'],
        cloud_df['senkou_span_b_shifted'],
        where=(cloud_df['senkou_span_a_shifted'] < cloud_df['senkou_span_b_shifted']),
        color='red',
        alpha=0.2
    )

    # === Optional: plot lines
    #ax.plot(df['date_num'], df['tenkan_sen'], label='Tenkan Sen', color='aqua', linestyle='--')
    ax.plot(df['date_num'], df['kijun_sen'], label='Kijun Sen', color='orange', linestyle='-')
    #ax.plot(df['date_num'], df['chikou_span'], label='Chikou Span', color='white', linestyle=':', alpha=0.6)

    # === Volume Bars ===
    volume = df['volume'].tail(60).values
    date_vol = df['date_num'].tail(60).values
    ax_vol.bar(date_vol, volume, width=0.12, color="#2f302f", alpha=0.6)
    ax_vol.set_facecolor("#fbfbfb")
    ax_vol.set_ylabel("Volume", fontsize=10)
    ax_vol.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    # === Candlestick Chart ===
    ohlc_data = df[['date_num', 'open', 'high', 'low', 'close']].tail(100).values
    candlestick_ohlc(ax, ohlc_data, width=0.12, colorup="#0e793a", colordown="#9d190a", alpha=0.9)

    # === Plot EMA lines & Supertrend
    ax.plot(df['date_num'], df['ema9'], label='EMA 9', color="#a8149e", linewidth=0.8, linestyle='-')
    ax.plot(df['date_num'], df['ema26'], label='EMA 26', color="#646464", linewidth=0.8, linestyle='-')
    ax.plot(df['date_num'], df['supertrend_up'], label='Supertrend Up', color="#05843A", linewidth=1.6)
    ax.plot(df['date_num'], df['supertrend_down'], label='Supertrend Down', color='red', linewidth=1.6)



    # === Stats Box (Latest candle - 1) ===
    idx = df.index[-2]
    stats = (
        f"Date: {df.loc[idx, 'datetime'].strftime('%Y-%m-%d %H:%M')}\n"
        f"Close: {df.loc[idx, 'close']:.2f} USDT\n"
        f"EMA9: {df.loc[idx, 'ema9']:.2f}\n"
        f"EMA26: {df.loc[idx, 'ema26']:.2f}\n"
        f"RSI: {df.loc[idx, 'rsi']:.2f}\n"
        f"RSI2: {df.loc[idx, 'rsi2']:.2f}\n"
        f"MFI: {df.loc[idx, 'mfi']:.2f}"
    )
    ax.text(0.02, 0.97, stats, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                                               facecolor='white',
                                               edgecolor='#bdc3c7',
                                               alpha=0.95),
            fontfamily='monospace')

    # === Axis & Aesthetics ===
    ax.set_xlim(left=df['date_num'].iloc[0], right=cloud_df['date_num'].iloc[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
    ax.legend(loc='upper right', fontsize=11, frameon=False)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_facecolor("#fbfbfb")
    ax.set_title(f"{symbol} – 4H Chart with EMA & Indicators", fontsize=18, weight='bold', color='#2c3e50')

    # === Save & Send ===
    output_file = f"{symbol}_signal_chart.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    send_chart_to_discord(output_file, webhook_url=DISCORD_WEBHOOK_URL)

# === EXCEPTIONAL PAIRS TO IGNORE ===
EXCLUDED_PAIRS = [
    "USDCUSDT", "BUSDUSDT", "FDUSDUSDT", "EURUSDT", "PAXGUSDT",  # Stablecoins or illiquid
    "1000SATSUSDT", "TRUMPUSDT", "MUBARAKUSDT", "SYRUPUSDT",     # Meme or low-volume tokens
    "OMUSDT","NEIROUSDT","PNUTUSDT","WBTCUSDT","RUNEUSDT", "BNSOLUSDT", # Shitcoin
]  
  
# === MAIN SCANNER ===
def run_screener():
    btc_price = fetch_latest_price()
    if btc_price:
        #send_telegram(f"📈 BTC/USDT Current Price: ${btc_price:,.2f}")
        send_discord(f"📈 BTC/USDT Current Price: ${btc_price:,.2f}")
        plot_signal_chart(symbol='BTCUSDT')
    else:
        #send_telegram("⚠️ Failed to fetch BTC/USDT price.")
        send_discord("⚠️ Failed to fetch BTC/USDT price.")
    symbols = get_top_100_usdt_symbols()
    for symbol in symbols:
        if symbol in EXCLUDED_PAIRS:
            print(f"⛔ Skipping {symbol} (excluded)")
            continue

        try:
            df = fetch_klines(symbol)
            if is_buy1_signal(df):
                entry_price = df.iloc[-2]['close']  # Use -2 to avoid current open bar
                #send_telegram(f"✅ BUY1 Trend Signal for {symbol}!\n🎯 Entry Price: ${entry_price:,.4f}")
                send_discord(f"✅ BUY1 Trend Signal for {symbol}!\n🎯 Entry Price: ${entry_price:,.4f}")
                plot_signal_chart(symbol=symbol)

            if is_buy2_signal(df):
                entry_price = df.iloc[-2]['close']  # Use -2 to avoid current open bar
                #send_telegram(f"✅ Dip Buy2 Signal for {symbol}!\n🎯 Entry Price: ${entry_price:,.4f}")
                send_discord(f"✅ Dip Buy2 Signal for {symbol}!\n🎯 Entry Price: ${entry_price:,.4f}")
                plot_signal_chart(symbol=symbol)
            
            if is_buy3_signal(df):
                entry_price = df.iloc[-2]['close']  # Use -2 to avoid current open bar
                #send_telegram(f"✅ Dip Buy3 Signal for {symbol}!\n🎯 Entry Price: ${entry_price:,.4f}")
                send_discord(f"✅ Dip Buy3 Signal for {symbol}!\n🎯 Entry Price: ${entry_price:,.4f}")
                plot_signal_chart(symbol=symbol)

            if symbol in ['BTCUSDT', 'ETHUSDT']:
                sell_trade_count.setdefault(symbol, 0)
                if is_sell_signal(df, sell_trade_count[symbol]):
                    if sell_trade_count[symbol] % 10 == 0:
                        entry_price = df.iloc[-2]['close']
                        #send_telegram(f"❌ SELL Signal for {symbol}\n🎯 Entry: {entry_price:,.2f}")
                        send_discord(f"❌ SELL Signal for {symbol}\n🎯 Entry: {entry_price:,.2f}")
                    sell_trade_count[symbol] += 1

        except Exception as e:
            print(f"Error with {symbol}: {e}")


# === SCHEDULER LOOP ===
def get_seconds_to_next_bar(interval_str):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    if interval_str.endswith('h'):
        hours = int(interval_str[:-1])
        current_hour = now.hour
        next_bar_hour = ((current_hour // hours) + 1) * hours

        if next_bar_hour >= 24:
            next_close = now.replace(hour=0, minute=0) + timedelta(days=1)
        else:
            next_close = now.replace(hour=next_bar_hour, minute=0)

    elif interval_str.endswith('d'):
        next_close = now.replace(hour=0, minute=0) + timedelta(days=1)

    elif interval_str.endswith('m'):
        minutes = int(interval_str[:-1])
        current_minute = now.minute
        next_bar_minute = ((current_minute // minutes) + 1) * minutes

        if next_bar_minute >= 60:
            next_close = now.replace(minute=0) + timedelta(hours=1)
        else:
            next_close = now.replace(minute=next_bar_minute)

    else:
        raise ValueError("Unsupported interval format. Use like '4h', '1d', '15m'.")

    # Ensure result is non-negative
    seconds_to_wait = max(0, int((next_close - now).total_seconds()))
    return seconds_to_wait

while True:
    wait_seconds = get_seconds_to_next_bar(TIMEFRAME)
    print(f"⏳ Waiting {wait_seconds:.0f} seconds until next {TIMEFRAME} bar close...")
    time.sleep(max(0, wait_seconds))
    print("▶️ Running scanner now...")
    run_screener()


    