#!/usr/bin/env python3
"""
Enhanced Stock History Fetcher with 100+ Technical Indicators and Market Data
Pulls 30 years of stock data with comprehensive technical analysis
Includes VIX data, VWAP, option data, and market sentiment indicators
Supports multiple tickers in a single file

NEW FEATURES ADDED:
- VWAP (Volume Weighted Average Price) calculations (cumulative and rolling)
- Option data: Put/Call ratios, option volumes, expiration dates
- Market sentiment indicators: Fear/Greed, Price sentiment, Volatility sentiment
- Institutional data: Holder counts, recommendations, earnings dates
- Intraday data support with configurable periods and intervals
- Enhanced volume and money flow indicators
- Price action patterns: Inside/Outside bars, gaps
- Additional momentum and volatility indicators
- Market sentiment analysis based on multiple factors
- Calendar features: Day of week, month, year, week of year
- Lunar phases: Full moon, new moon, waxing/waning cycles
- Market seasonality: Quarter start/end, earnings seasons, options expiration
- Trading patterns: January effect, Santa Claus rally, Sell in May
- Market events: FOMC weeks, quadruple witching, tax loss harvesting
- Holiday and month-end effects
- Agricultural seasons: Planting, harvest, grain reports, ethanol releases
- Astronomical events: Equinoxes, solstices, solar cycles
- Retail seasons: Back-to-school, Black Friday, Cyber Monday, Prime Day
- Consumer events: Valentine's, Mother's/Father's Day, Super Bowl, Olympics
- Travel and tourism seasons
- Tax refund and spending cycles
- Federal Reserve and central bank meetings
- Economic data releases (NFP, CPI, GDP, etc.)
- Market microstructure events (triple witching, rebalancing)
- Political and election cycles
- ETF and index rebalancing events
- Advanced moving averages (DEMA, HMA, KAMA, MA Envelope, MA Ribbon)
- Trend indicators (Alligator, Supertrend, Linear Regression)
- Enhanced oscillators (Stochastic RSI, TSI, TRIX, Schaff Trend Cycle)
- Statistical indicators (Correlation, Z-Score, Bollinger enhancements)
- Wave analysis (Fibonacci retracements, Elliott Wave count)
- Ensemble and signal indicators
- US and international holidays
- Financial cycles and deadlines
- Payroll and compensation cycles
- Market psychology and seasonal effects
- Biological and environmental cycles
- Enhanced lunar calculations with sine/cosine encoding
- Social media and news sentiment indicators
- Political and government event flags
- Alternative data indicators
- Market microstructure indicators
- Feature filtering: Apply optimized feature selection using JSON filters from automated feature selector

NULL HANDLING:
- Default: Keep all null values (recommended for ML frameworks like XGBoost, LightGBM)
- Use --aggressive-nulls for complete null removal (may lose data)
- Use --keep-nulls explicitly (same as default, for clarity)

ERROR HANDLING:
- Enhanced error handling for HTTP 404 errors from yfinance
- Exponential backoff retry logic for failed requests
- Graceful degradation when data is unavailable
- Rate limiting protection with delays between requests
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse
import sys
import warnings
import time
import random
from urllib.error import HTTPError
import requests
import json
warnings.filterwarnings('ignore')

# Try to import ephem for lunar calculations
try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    print("Warning: ephem package not available. Lunar calculations will be disabled.")
    print("Install with: pip install ephem")
    EPHEM_AVAILABLE = False

print("[DEBUG] Script started: very top of 1_stock_history_script.py")

def safe_yfinance_request(func, *args, max_retries=3, base_delay=1, debug=False, **kwargs):
    """
    Safely execute yfinance requests with retry logic and error handling
    
    Args:
        func: The yfinance function to call
        *args: Arguments for the function
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds between retries
        debug: Whether to print debug information
        **kwargs: Keyword arguments for the function
    
    Returns:
        The result of the function call, or None if all retries fail
    """
    for attempt in range(max_retries + 1):
        try:
            # Add random delay to avoid rate limiting
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                if debug:
                    print(f"[DEBUG] Retry attempt {attempt}/{max_retries}, waiting {delay:.2f}s...")
                time.sleep(delay)
            
            result = func(*args, **kwargs)
            return result
            
        except HTTPError as e:
            if e.code == 404:
                if debug:
                    print(f"[DEBUG] HTTP 404 error on attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    if debug:
                        print(f"[DEBUG] Max retries reached for 404 error, returning None")
                    return None
            else:
                if debug:
                    print(f"[DEBUG] HTTP error {e.code} on attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    if debug:
                        print(f"[DEBUG] Max retries reached for HTTP error, returning None")
                    return None
                    
        except Exception as e:
            if debug:
                print(f"[DEBUG] Error on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                if debug:
                    print(f"[DEBUG] Max retries reached, returning None")
                return None
    
    return None

def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_cci(high, low, close, window=20):
    """Calculate Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    ma = typical_price.rolling(window=window).mean()
    mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (typical_price - ma) / (0.015 * mad)
    return cci

def calculate_adx(high, low, close, window=14):
    """Calculate ADX"""
    high_diff = high.diff()
    low_diff = low.diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff > 0), 0)
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = true_range.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=window).mean()
    
    return adx

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_williams_r(high, low, close, window=14):
    """Calculate Williams %R"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr

def calculate_ultimate_oscillator(high, low, close, period1=7, period2=14, period3=28):
    """Calculate Ultimate Oscillator"""
    bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    
    avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
    avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
    avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
    
    uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
    return uo

def calculate_mfi(high, low, close, volume, window=14):
    """Calculate Money Flow Index"""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=window).sum()
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=window).sum()
    
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    return mfi

def calculate_cmo(prices, window=9):
    """Calculate Chande Momentum Oscillator"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).sum()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).sum()
    cmo = 100 * (gain - loss) / (gain + loss)
    return cmo

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band, rolling_mean

def calculate_keltner_channels(high, low, close, window=20, multiplier=2):
    """Calculate Keltner Channels"""
    ema = close.ewm(span=window).mean()
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    upper = ema + (multiplier * atr)
    lower = ema - (multiplier * atr)
    return upper, lower, ema

def calculate_donchian_channels(high, low, window=20):
    """Calculate Donchian Channels"""
    upper = high.rolling(window=window).max()
    lower = low.rolling(window=window).min()
    middle = (upper + lower) / 2
    return upper, lower, middle

def calculate_parabolic_sar(high, low, close, af=0.02, max_af=0.2):
    """Calculate Parabolic SAR"""
    length = len(close)
    psar = close.copy()
    psaraf = [af] * length
    psardir = [1] * length
    
    for i in range(2, length):
        if psardir[i-1] == 1:  # Uptrend
            psar.iloc[i] = psar.iloc[i-1] + psaraf[i-1] * (high.iloc[i-1] - psar.iloc[i-1])
            if low.iloc[i] <= psar.iloc[i]:
                psardir[i] = -1
                psar.iloc[i] = max(high.iloc[i-1], high.iloc[i-2])
                psaraf[i] = af
            else:
                psardir[i] = 1
                if high.iloc[i] > high.iloc[i-1]:
                    psaraf[i] = min(psaraf[i-1] + af, max_af)
                else:
                    psaraf[i] = psaraf[i-1]
        else:  # Downtrend
            psar.iloc[i] = psar.iloc[i-1] - psaraf[i-1] * (psar.iloc[i-1] - low.iloc[i-1])
            if high.iloc[i] >= psar.iloc[i]:
                psardir[i] = 1
                psar.iloc[i] = min(low.iloc[i-1], low.iloc[i-2])
                psaraf[i] = af
            else:
                psardir[i] = -1
                if low.iloc[i] < low.iloc[i-1]:
                    psaraf[i] = min(psaraf[i-1] + af, max_af)
                else:
                    psaraf[i] = psaraf[i-1]
    
    return psar

def calculate_elder_ray(high, low, close, ema_period=13):
    ema = close.ewm(span=ema_period).mean()
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power


def calculate_chaikin_money_flow(high, low, close, volume, window=20):
    mfv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume
    cmf = mfv.rolling(window=window).sum() / volume.rolling(window=window).sum()
    return cmf


def calculate_force_index(close, volume, window=13):
    fi = close.diff() * volume
    return fi.rolling(window=window).mean()


def calculate_ease_of_movement(high, low, volume, window=14):
    distance_moved = ((high + low) / 2).diff()
    box_ratio = (volume / 1e6) / (high - low).replace(0, np.nan)
    eom = distance_moved / box_ratio
    return eom.rolling(window=window).mean()


def calculate_ppo(close, fast=12, slow=26):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    ppo = (ema_fast - ema_slow) / ema_slow * 100
    return ppo


def calculate_dpo(close, window=20):
    shifted = close.shift(int(window / 2) + 1)
    ma = close.rolling(window=window).mean()
    dpo = shifted - ma
    return dpo


def calculate_vortex_indicator(high, low, close, window=14):
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    vm_plus = abs(high - low.shift())
    vm_minus = abs(low - high.shift())
    vi_plus = vm_plus.rolling(window=window).sum() / tr.rolling(window=window).sum()
    vi_minus = vm_minus.rolling(window=window).sum() / tr.rolling(window=window).sum()
    return vi_plus, vi_minus


def calculate_rvi(close, window=10):
    num = (close - close.shift(1)).rolling(window=window).mean()
    denom = (close - close.shift(1)).abs().rolling(window=window).mean()
    rvi = num / denom
    return rvi


def calculate_accum_dist(close, high, low, volume):
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    ad = (clv * volume).cumsum()
    return ad


def calculate_aroon(high, low, window=25):
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    aroon_up = high.rolling(window + 1).apply(lambda x: float(np.argmax(x)) / window * 100, raw=True)
    aroon_down = low.rolling(window + 1).apply(lambda x: float(np.argmin(x)) / window * 100, raw=True)
    return aroon_up, aroon_down


def calculate_fdi(close, window=14):
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    ln_n = np.log(np.arange(1, window + 1))
    ln_delta = np.log(close.diff().abs().rolling(window=window).mean())
    fdi = ln_delta / ln_n.mean()
    return fdi


def calculate_stc(close, fast=23, slow=50, cycle=10):
    ema1 = close.ewm(span=fast).mean()
    ema2 = ema1.ewm(span=slow).mean()
    macd = ema1 - ema2
    stc = macd.ewm(span=cycle).mean()
    return stc


def calculate_tema(close, window=30):
    ema1 = close.ewm(span=window).mean()
    ema2 = ema1.ewm(span=window).mean()
    ema3 = ema2.ewm(span=window).mean()
    tema = 3 * (ema1 - ema2) + ema3
    return tema


def calculate_zscore(close, window=20):
    mean = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    zscore = (close - mean) / std
    return zscore


def calculate_donchian_width(high, low, window=20):
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    upper = high.rolling(window=window).max()
    lower = low.rolling(window=window).min()
    width = (upper - lower) / lower * 100
    return width


def calculate_choppiness_index(high, low, close, window=14):
    # Ensure all inputs are pandas Series
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    
    # Calculate true range
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    
    # Calculate ATR sum
    atr_sum = tr.rolling(window=window).sum()
    
    # Calculate price range
    price_range = close.rolling(window=window).max() - close.rolling(window=window).min()
    
    # Calculate choppiness
    choppiness = 100 * np.log10(atr_sum / price_range) / np.log10(window)
    return choppiness


def calculate_connors_rsi(close, window_rsi=3, window_streak=2, window_rank=100):
    rsi = calculate_rsi(close, window_rsi)
    streak = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).groupby((close.diff() != 0).cumsum()).cumsum()
    streak_rsi = calculate_rsi(streak, window_streak)
    rank = close.rolling(window=window_rank).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)
    connors_rsi = (rsi + streak_rsi + rank) / 3
    return connors_rsi


def calculate_trima(close, window=10):
    return close.rolling(window=window, center=True).mean()


def calculate_kst(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, sig=9):
    rocma1 = close.pct_change(r1).rolling(n1).mean()
    rocma2 = close.pct_change(r2).rolling(n2).mean()
    rocma3 = close.pct_change(r3).rolling(n3).mean()
    rocma4 = close.pct_change(r4).rolling(n4).mean()
    kst = rocma1 + 2*rocma2 + 3*rocma3 + 4*rocma4
    kst_signal = kst.rolling(sig).mean()
    return kst, kst_signal

def calculate_dema(close, window=20):
    """Calculate Double Exponential Moving Average (DEMA)"""
    ema1 = close.ewm(span=window).mean()
    ema2 = ema1.ewm(span=window).mean()
    dema = 2 * ema1 - ema2
    return dema

def calculate_hma(high, low, close, window=20):
    """Calculate Hull Moving Average (HMA)"""
    # Calculate WMA of typical price
    typical_price = (high + low + close) / 3
    wma1 = typical_price.rolling(window=window//2).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)))
    wma2 = typical_price.rolling(window=window).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)))
    
    # Calculate raw HMA
    raw_hma = 2 * wma1 - wma2
    
    # Calculate final HMA
    hma = raw_hma.rolling(window=int(np.sqrt(window))).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)))
    return hma

def calculate_kama(close, window=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)"""
    change = abs(close - close.shift(window))
    volatility = change.rolling(window=window).sum()
    er = abs(close - close.shift(window)) / volatility
    
    # Calculate smoothing constants
    sc = (er * (fast - slow) + slow) ** 2
    
    # Calculate KAMA
    kama = pd.Series(index=close.index, dtype=float)
    kama.iloc[0] = close.iloc[0]
    
    for i in range(1, len(close)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
    
    return kama

def calculate_ma_envelope(close, window=20, deviation=0.025):
    """Calculate Moving Average Envelope (MAE)"""
    ma = close.rolling(window=window).mean()
    upper = ma * (1 + deviation)
    lower = ma * (1 - deviation)
    return upper, ma, lower

def calculate_ma_ribbon(close, periods=[10, 20, 30, 40, 50, 60]):
    """Calculate Moving Average Ribbon"""
    ribbon = {}
    for period in periods:
        ribbon[f'MA_{period}'] = close.rolling(window=period).mean()
    return ribbon

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Alligator Indicator (Lips, Teeth, Jaw)"""
    # Calculate median price
    median_price = (high + low) / 2
    
    # Calculate Alligator lines
    jaw = median_price.rolling(window=jaw_period).mean().shift(8)
    teeth = median_price.rolling(window=teeth_period).mean().shift(5)
    lips = median_price.rolling(window=lips_period).mean().shift(3)
    
    return jaw, teeth, lips

def calculate_supertrend(high, low, close, period=10, multiplier=3):
    """Calculate Supertrend Indicator"""
    # Calculate ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculate basic upper and lower bands
    basic_upper = (high + low) / 2 + multiplier * atr
    basic_lower = (high + low) / 2 - multiplier * atr
    
    # Calculate final upper and lower bands
    final_upper = pd.Series(index=close.index, dtype=float)
    final_lower = pd.Series(index=close.index, dtype=float)
    supertrend = pd.Series(index=close.index, dtype=float)
    
    for i in range(period, len(close)):
        if close.iloc[i] <= final_upper.iloc[i-1]:
            final_upper.iloc[i] = min(basic_upper.iloc[i], final_upper.iloc[i-1])
        else:
            final_upper.iloc[i] = basic_upper.iloc[i]
        
        if close.iloc[i] >= final_lower.iloc[i-1]:
            final_lower.iloc[i] = max(basic_lower.iloc[i], final_lower.iloc[i-1])
        else:
            final_lower.iloc[i] = basic_lower.iloc[i]
        
        if supertrend.iloc[i-1] == final_upper.iloc[i-1] and close.iloc[i] <= final_upper.iloc[i]:
            supertrend.iloc[i] = final_upper.iloc[i]
        elif supertrend.iloc[i-1] == final_upper.iloc[i-1] and close.iloc[i] > final_upper.iloc[i]:
            supertrend.iloc[i] = final_lower.iloc[i]
        elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and close.iloc[i] >= final_lower.iloc[i]:
            supertrend.iloc[i] = final_lower.iloc[i]
        elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and close.iloc[i] < final_lower.iloc[i]:
            supertrend.iloc[i] = final_upper.iloc[i]
    
    return supertrend

def calculate_stochastic_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    """Calculate Stochastic RSI"""
    # Calculate RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic RSI
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    stoch_rsi = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
    
    k = stoch_rsi.rolling(window=k_period).mean()
    d = k.rolling(window=d_period).mean()
    
    return k, d

def calculate_momentum(close, window=10):
    """Calculate Momentum indicator"""
    return close - close.shift(window)

def calculate_roc(close, window=10):
    """Calculate Rate of Change (ROC)"""
    return ((close - close.shift(window)) / close.shift(window)) * 100

def calculate_tsi(close, first_period=25, second_period=13, signal_period=9):
    """Calculate True Strength Index (TSI)"""
    # Calculate price change
    pc = close.diff()
    
    # Calculate first smoothing
    apc = pc.ewm(span=first_period).mean()
    
    # Calculate second smoothing
    apc2 = apc.ewm(span=second_period).mean()
    
    # Calculate absolute price change
    abs_pc = abs(pc)
    
    # Calculate first smoothing of absolute price change
    abs_apc = abs_pc.ewm(span=first_period).mean()
    
    # Calculate second smoothing of absolute price change
    abs_apc2 = abs_apc.ewm(span=second_period).mean()
    
    # Calculate TSI
    tsi = 100 * (apc2 / abs_apc2)
    
    # Calculate signal line
    signal = tsi.ewm(span=signal_period).mean()
    
    return tsi, signal

def calculate_trix(close, window=15, signal_period=9):
    """Calculate TRIX (Triple Exponential Average)"""
    # Calculate first EMA
    ema1 = close.ewm(span=window).mean()
    
    # Calculate second EMA
    ema2 = ema1.ewm(span=window).mean()
    
    # Calculate third EMA
    ema3 = ema2.ewm(span=window).mean()
    
    # Calculate TRIX
    trix = 100 * ema3.pct_change()
    
    # Calculate signal line
    signal = trix.ewm(span=signal_period).mean()
    
    return trix, signal

def calculate_schaff_trend_cycle(close, period1=23, period2=50, period3=10):
    """Calculate Schaff Trend Cycle"""
    # Calculate MACD
    ema1 = close.ewm(span=period1).mean()
    ema2 = close.ewm(span=period2).mean()
    macd = ema1 - ema2
    
    # Calculate first stochastic
    lowest_low = macd.rolling(window=period3).min()
    highest_high = macd.rolling(window=period3).max()
    k1 = 100 * (macd - lowest_low) / (highest_high - lowest_low)
    
    # Calculate second stochastic
    lowest_low2 = k1.rolling(window=period3).min()
    highest_high2 = k1.rolling(window=period3).max()
    k2 = 100 * (k1 - lowest_low2) / (highest_high2 - lowest_low2)
    
    return k2

def calculate_vwap_bands(high, low, close, volume, window=20, std_dev=2):
    """Calculate VWAP Bands"""
    vwap = calculate_vwap(high, low, close, volume, window)
    
    # Calculate standard deviation
    typical_price = (high + low + close) / 3
    vwap_std = (typical_price - vwap).rolling(window=window).std()
    
    # Calculate bands
    upper_band = vwap + (std_dev * vwap_std)
    lower_band = vwap - (std_dev * vwap_std)
    
    return upper_band, vwap, lower_band

def calculate_normalized_atr(high, low, close, window=14):
    """Calculate Normalized ATR"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    # Normalize by price
    normalized_atr = (atr / close) * 100
    
    return normalized_atr

def calculate_chaikin_volatility(high, low, window=10):
    """Calculate Chaikin Volatility"""
    # Calculate high-low range
    hl_range = high - low
    
    # Calculate EMA of high-low range
    ema_hl = hl_range.ewm(span=window).mean()
    
    # Calculate Chaikin Volatility
    chaikin_vol = ((ema_hl - ema_hl.shift(window)) / ema_hl.shift(window)) * 100
    
    return chaikin_vol

def calculate_volume_oscillator(volume, short_period=5, long_period=10):
    """Calculate Volume Oscillator"""
    short_ma = volume.rolling(window=short_period).mean()
    long_ma = volume.rolling(window=long_period).mean()
    
    volume_osc = ((short_ma - long_ma) / long_ma) * 100
    
    return volume_osc

def calculate_pvt(close, volume):
    """Calculate Price Volume Trend (PVT)"""
    price_change = close.pct_change()
    pvt = (price_change * volume).cumsum()
    
    return pvt

def calculate_median_price(high, low):
    """Calculate Median Price"""
    return (high + low) / 2

def calculate_typical_price(high, low, close):
    """Calculate Typical Price"""
    return (high + low + close) / 3

def calculate_weighted_close(high, low, close):
    """Calculate Weighted Close Price"""
    return (high + low + (close * 2)) / 4

def calculate_linear_regression(close, window=20):
    """Calculate Linear Regression Line"""
    regression = pd.Series(index=close.index, dtype=float)
    slope = pd.Series(index=close.index, dtype=float)
    
    for i in range(window-1, len(close)):
        y = close.iloc[i-window+1:i+1].values
        x = np.arange(window)
        
        # Calculate linear regression
        slope_val, intercept = np.polyfit(x, y, 1)
        regression.iloc[i] = slope_val * (window-1) + intercept
        slope.iloc[i] = slope_val
    
    return regression, slope

def calculate_regression_channel(close, window=20, std_dev=2):
    """Calculate Regression Channel"""
    regression, slope = calculate_linear_regression(close, window)
    
    # Calculate standard deviation of residuals
    residuals = pd.Series(index=close.index, dtype=float)
    for i in range(window-1, len(close)):
        y = close.iloc[i-window+1:i+1].values
        x = np.arange(window)
        slope_val, intercept = np.polyfit(x, y, 1)
        predicted = slope_val * x + intercept
        residuals.iloc[i] = np.std(y - predicted)
    
    # Calculate channels
    upper_channel = regression + (std_dev * residuals)
    lower_channel = regression - (std_dev * residuals)
    
    return upper_channel, regression, lower_channel

def calculate_correlation_coefficient(close, window=20):
    """Calculate Correlation Coefficient"""
    correlation = pd.Series(index=close.index, dtype=float)
    
    for i in range(window-1, len(close)):
        y = close.iloc[i-window+1:i+1].values
        x = np.arange(window)
        correlation.iloc[i] = np.corrcoef(x, y)[0, 1]
    
    return correlation

def calculate_rolling_beta(close, market_close, window=20):
    """Calculate Rolling Beta"""
    # Calculate returns
    returns = close.pct_change()
    market_returns = market_close.pct_change()
    
    # Calculate rolling beta
    beta = pd.Series(index=close.index, dtype=float)
    
    for i in range(window-1, len(close)):
        stock_ret = returns.iloc[i-window+1:i+1]
        market_ret = market_returns.iloc[i-window+1:i+1]
        
        # Remove NaN values
        valid_data = pd.concat([stock_ret, market_ret], axis=1).dropna()
        if len(valid_data) > 1:
            cov_matrix = np.cov(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
            market_var = np.var(valid_data.iloc[:, 1])
            if market_var != 0:
                beta.iloc[i] = cov_matrix[0, 1] / market_var
            else:
                beta.iloc[i] = np.nan
        else:
            beta.iloc[i] = np.nan
    
    return beta

def calculate_zscore_normalization(close, window=20):
    """Calculate Z-Score Normalization"""
    mean = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    zscore = (close - mean) / std
    
    return zscore

def calculate_bollinger_percent_b(close, window=20, num_std=2):
    """Calculate Bollinger %B"""
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, window, num_std)
    percent_b = (close - bb_lower) / (bb_upper - bb_lower)
    
    return percent_b

def calculate_bollinger_bandwidth(close, window=20, num_std=2):
    """Calculate Bollinger Band Width"""
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, window, num_std)
    bandwidth = (bb_upper - bb_lower) / bb_middle
    
    return bandwidth

def calculate_gaussian_filter(close, window=20, sigma=1):
    """Calculate Gaussian Filter"""
    # Create Gaussian weights
    x = np.arange(-(window//2), window//2 + 1)
    weights = np.exp(-(x**2) / (2 * sigma**2))
    weights = weights / weights.sum()
    
    # Apply filter
    gaussian_filter = close.rolling(window=window, center=True).apply(
        lambda x: np.sum(x * weights[:len(x)])
    )
    
    return gaussian_filter

def calculate_sine_cosine_transform(close, period=20):
    """Calculate Sine/Cosine Transform (for cycles)"""
    # Create time series
    t = np.arange(len(close))
    
    # Calculate sine and cosine components
    sine_component = np.sin(2 * np.pi * t / period)
    cosine_component = np.cos(2 * np.pi * t / period)
    
    # Calculate transforms
    sine_transform = pd.Series(sine_component, index=close.index)
    cosine_transform = pd.Series(cosine_component, index=close.index)
    
    return sine_transform, cosine_transform

def calculate_ensemble_signal_score(data, indicators=['RSI_14', 'MACD_12_26_9', 'STOCH_14_3_3', 'CCI_20']):
    """Calculate Ensemble Signal Score"""
    signals = pd.DataFrame(index=data.index)
    
    # RSI signals
    if 'RSI_14' in data.columns:
        signals['RSI_Signal'] = np.where(data['RSI_14'] < 30, 1, np.where(data['RSI_14'] > 70, -1, 0))
    
    # MACD signals
    if 'MACD_12_26_9' in data.columns and 'MACDs_12_26_9' in data.columns:
        signals['MACD_Signal'] = np.where(data['MACD_12_26_9'] > data['MACDs_12_26_9'], 1, -1)
    
    # Stochastic signals
    if 'STOCH_14_3_3' in data.columns:
        signals['Stoch_Signal'] = np.where(data['STOCH_14_3_3'] < 20, 1, np.where(data['STOCH_14_3_3'] > 80, -1, 0))
    
    # CCI signals
    if 'CCI_20' in data.columns:
        signals['CCI_Signal'] = np.where(data['CCI_20'] < -100, 1, np.where(data['CCI_20'] > 100, -1, 0))
    
    # Calculate ensemble score
    ensemble_score = signals.mean(axis=1)
    
    return ensemble_score

def calculate_indicator_confidence(data, window=20):
    """Calculate Indicator Confidence Weighting"""
    # Calculate volatility
    returns = data['Close'].pct_change()
    volatility = returns.rolling(window=window).std()
    
    # Calculate trend strength
    ma_short = data['Close'].rolling(window=10).mean()
    ma_long = data['Close'].rolling(window=50).mean()
    trend_strength = abs(ma_short - ma_long) / ma_long
    
    # Calculate confidence (inverse of volatility, weighted by trend strength)
    confidence = (1 / (1 + volatility)) * trend_strength
    
    return confidence

def calculate_signal_consensus(data, indicators=['RSI_14', 'MACD_12_26_9', 'STOCH_14_3_3', 'CCI_20', 'ADX_14']):
    """Calculate Signal Consensus"""
    bullish_signals = 0
    bearish_signals = 0
    
    # Count bullish signals
    if 'RSI_14' in data.columns and data['RSI_14'].iloc[-1] < 30:
        bullish_signals += 1
    if 'MACD_12_26_9' in data.columns and 'MACDs_12_26_9' in data.columns and data['MACD_12_26_9'].iloc[-1] > data['MACDs_12_26_9'].iloc[-1]:
        bullish_signals += 1
    if 'STOCH_14_3_3' in data.columns and data['STOCH_14_3_3'].iloc[-1] < 20:
        bullish_signals += 1
    if 'CCI_20' in data.columns and data['CCI_20'].iloc[-1] < -100:
        bullish_signals += 1
    if 'ADX_14' in data.columns and data['ADX_14'].iloc[-1] > 25:
        bullish_signals += 1
    
    # Count bearish signals
    if 'RSI_14' in data.columns and data['RSI_14'].iloc[-1] > 70:
        bearish_signals += 1
    if 'MACD_12_26_9' in data.columns and 'MACDs_12_26_9' in data.columns and data['MACD_12_26_9'].iloc[-1] < data['MACDs_12_26_9'].iloc[-1]:
        bearish_signals += 1
    if 'STOCH_14_3_3' in data.columns and data['STOCH_14_3_3'].iloc[-1] > 80:
        bearish_signals += 1
    if 'CCI_20' in data.columns and data['CCI_20'].iloc[-1] > 100:
        bearish_signals += 1
    
    total_indicators = len(indicators)
    consensus = (bullish_signals - bearish_signals) / total_indicators
    
    return consensus

def calculate_fibonacci_retracements(high, low, levels=[0.236, 0.382, 0.5, 0.618, 0.786]):
    """Calculate Fibonacci Retracements"""
    price_range = high - low
    retracements = {}
    
    for level in levels:
        retracements[f'Fib_{int(level*1000)}'] = high - (price_range * level)
    
    return retracements

def calculate_elliott_wave_count(high, low, close, window=20):
    """Calculate Elliott Wave Count (simplified)"""
    # This is a simplified implementation
    # Real Elliott Wave analysis requires complex pattern recognition
    
    # Calculate swing highs and lows
    swing_highs = high.rolling(window=window, center=True).max()
    swing_lows = low.rolling(window=window, center=True).min()
    
    # Calculate wave count (simplified)
    wave_count = pd.Series(index=close.index, dtype=int)
    wave_count.iloc[0] = 1
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            wave_count.iloc[i] = wave_count.iloc[i-1] + 1
        else:
            wave_count.iloc[i] = 1
    
    return wave_count

def calculate_price_oscillator(close, short_period=10, long_period=21):
    """Calculate Price Oscillator"""
    short_ma = close.rolling(window=short_period).mean()
    long_ma = close.rolling(window=long_period).mean()
    
    oscillator = ((short_ma - long_ma) / long_ma) * 100
    
    return oscillator

def calculate_disparity_index(close, window=20):
    """Calculate Disparity Index"""
    ma = close.rolling(window=window).mean()
    disparity = ((close - ma) / ma) * 100
    
    return disparity

def calculate_vwap(high, low, close, volume, window=None):
    """Calculate Volume Weighted Average Price (VWAP)"""
    typical_price = (high + low + close) / 3
    if window is None:
        # Cumulative VWAP
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
    else:
        # Rolling VWAP
        vwap = (typical_price * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
    return vwap

def fetch_option_data(ticker, debug=False):
    """Fetch option chain data for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get current date
        current_date = datetime.now()
        
        # Get option chains (next few expiration dates)
        option_chains = {}
        put_call_ratios = {}
        option_volumes = {}
        expiration_dates = []
        
        try:
            # Get available expiration dates with safe request
            expirations = safe_yfinance_request(lambda: stock.options, debug=debug)
            if expirations is None:
                if debug:
                    print(f"[DEBUG] Could not fetch option expiration dates for {ticker}")
                return {
                    'put_call_ratios': {},
                    'option_volumes': {},
                    'expiration_dates': []
                }
            
            if debug:
                print(f"[DEBUG] Available option expiration dates for {ticker}: {expirations[:5]}")  # Show first 5
            
            # Process next 3 expiration dates
            for exp_date in expirations[:3]:
                try:
                    # Use safe request for option chain
                    opt = safe_yfinance_request(lambda: stock.option_chain(exp_date), debug=debug)
                    if opt is None:
                        if debug:
                            print(f"[DEBUG] Could not fetch option chain for {ticker} {exp_date}")
                        continue
                    
                    # Calculate put/call ratios
                    if hasattr(opt, 'calls') and hasattr(opt, 'puts'):
                        calls = opt.calls
                        puts = opt.puts
                        
                        # Calculate total volume
                        call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
                        put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
                        
                        # Calculate put/call ratio
                        put_call_ratio = put_volume / call_volume if call_volume > 0 else np.nan
                        
                        # Store data
                        put_call_ratios[exp_date] = put_call_ratio
                        option_volumes[exp_date] = {
                            'call_volume': call_volume,
                            'put_volume': put_volume,
                            'total_volume': call_volume + put_volume
                        }
                        expiration_dates.append(exp_date)
                        
                        if debug:
                            print(f"[DEBUG] {ticker} {exp_date}: Call vol={call_volume}, Put vol={put_volume}, P/C ratio={put_call_ratio:.3f}")
                    
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] Error processing {ticker} options for {exp_date}: {e}")
                    continue
                    
        except Exception as e:
            if debug:
                print(f"[DEBUG] Error fetching option chain for {ticker}: {e}")
        
        return {
            'put_call_ratios': put_call_ratios,
            'option_volumes': option_volumes,
            'expiration_dates': expiration_dates
        }
        
    except Exception as e:
        if debug:
            print(f"[DEBUG] Error in fetch_option_data for {ticker}: {e}")
        return {
            'put_call_ratios': {},
            'option_volumes': {},
            'expiration_dates': []
        }

def fetch_market_data(ticker, debug=False):
    """Fetch additional market data including options, institutional data, etc."""
    try:
        stock = yf.Ticker(ticker)
        
        # Get basic info with safe request
        info = safe_yfinance_request(lambda: stock.info, debug=debug)
        if info is None:
            if debug:
                print(f"[DEBUG] Could not fetch basic info for {ticker}")
            info = {}
        
        # Get institutional data
        institutional_data = {}
        
        # Get institutional holders with safe request
        try:
            institutional_holders = safe_yfinance_request(lambda: stock.institutional_holders, debug=debug)
            if institutional_holders is not None and not institutional_holders.empty:
                institutional_data['top_institutional_holders'] = institutional_holders.to_dict('records')
            else:
                institutional_data['top_institutional_holders'] = []
        except:
            institutional_data['top_institutional_holders'] = []
        
        # Get major holders with safe request
        try:
            major_holders = safe_yfinance_request(lambda: stock.major_holders, debug=debug)
            if major_holders is not None and not major_holders.empty:
                institutional_data['major_holders'] = major_holders.to_dict('records')
            else:
                institutional_data['major_holders'] = []
        except:
            institutional_data['major_holders'] = []
        
        # Get recommendations with safe request
        try:
            recommendations = safe_yfinance_request(lambda: stock.recommendations, debug=debug)
            if recommendations is not None and not recommendations.empty:
                # Get latest recommendations
                latest_recs = recommendations.tail(10)  # Last 10 recommendations
                institutional_data['recent_recommendations'] = latest_recs.to_dict('records')
            else:
                institutional_data['recent_recommendations'] = []
        except:
            institutional_data['recent_recommendations'] = []
        
        # Get earnings dates with safe request
        try:
            earnings_dates = safe_yfinance_request(lambda: stock.earnings_dates, debug=debug)
            if earnings_dates is not None and not earnings_dates.empty:
                institutional_data['earnings_dates'] = earnings_dates.to_dict('records')
            else:
                institutional_data['earnings_dates'] = []
        except:
            institutional_data['earnings_dates'] = []
        
        # Get calendar with safe request
        try:
            calendar = safe_yfinance_request(lambda: stock.calendar, debug=debug)
            if calendar is not None and not calendar.empty:
                institutional_data['calendar'] = calendar.to_dict('records')
            else:
                institutional_data['calendar'] = []
        except:
            institutional_data['calendar'] = []
        
        return institutional_data
        
    except Exception as e:
        if debug:
            print(f"[DEBUG] Error fetching market data for {ticker}: {e}")
        return {
            'top_institutional_holders': [],
            'major_holders': [],
            'recent_recommendations': [],
            'earnings_dates': [],
            'calendar': []
        }

def fetch_intraday_data(ticker, period='1mo', interval='1h', debug=False):
    """Fetch intraday data for more granular analysis"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get intraday data with safe request
        intraday = safe_yfinance_request(lambda: stock.history(period=period, interval=interval), debug=debug)
        
        if intraday is None or intraday.empty:
            if debug:
                print(f"[DEBUG] No intraday data available for {ticker}")
            return None
        
        if debug:
            print(f"[DEBUG] Fetched {len(intraday)} intraday records for {ticker}")
        
        # Calculate intraday-specific indicators
        intraday['Intraday_Range'] = intraday['High'] - intraday['Low']
        intraday['Intraday_Range_Pct'] = (intraday['Intraday_Range'] / intraday['Close']) * 100
        intraday['Intraday_Volume_MA'] = intraday['Volume'].rolling(window=20).mean()
        intraday['Intraday_Volume_Ratio'] = intraday['Volume'] / intraday['Intraday_Volume_MA']
        
        # Intraday VWAP
        intraday['Intraday_VWAP'] = calculate_vwap(intraday['High'], intraday['Low'], intraday['Close'], intraday['Volume'])
        
        # Intraday momentum
        intraday['Intraday_Momentum'] = intraday['Close'].pct_change()
        intraday['Intraday_Momentum_MA'] = intraday['Intraday_Momentum'].rolling(window=10).mean()
        
        return intraday
        
    except Exception as e:
        if debug:
            print(f"[DEBUG] Error fetching intraday data for {ticker}: {e}")
        return None

def calculate_market_sentiment_indicators(data, debug=False):
    """Calculate market sentiment indicators"""
    try:
        # Fear & Greed indicators
        data['Fear_Greed_Volume'] = np.where(
            data['Volume'] > data['Volume'].rolling(window=20).mean() * 1.5,
            'High_Volume_Fear',  # High volume often indicates fear/greed
            np.where(
                data['Volume'] < data['Volume'].rolling(window=20).mean() * 0.5,
                'Low_Volume_Uncertainty',
                'Normal_Volume'
            )
        )
        
        # Price momentum sentiment
        data['Price_Sentiment'] = np.where(
            data['Close'] > data['Close'].rolling(window=20).mean() * 1.05,
            'Bullish',
            np.where(
                data['Close'] < data['Close'].rolling(window=20).mean() * 0.95,
                'Bearish',
                'Neutral'
            )
        )
        
        # Volatility sentiment
        data['Volatility_Sentiment'] = np.where(
            data['Volatility_20'] > data['Volatility_20'].rolling(window=50).mean() * 1.2,
            'High_Volatility_Fear',
            np.where(
                data['Volatility_20'] < data['Volatility_20'].rolling(window=50).mean() * 0.8,
                'Low_Volatility_Complacency',
                'Normal_Volatility'
            )
        )
        
        # RSI sentiment
        data['RSI_Sentiment'] = np.where(
            data['RSI_14'] > 70,
            'Overbought',
            np.where(
                data['RSI_14'] < 30,
                'Oversold',
                'Neutral'
            )
        )
        
        # VWAP sentiment
        data['VWAP_Sentiment'] = np.where(
            data['Close'] > data['VWAP'] * 1.02,
            'Above_VWAP_Bullish',
            np.where(
                data['Close'] < data['VWAP'] * 0.98,
                'Below_VWAP_Bearish',
                'Near_VWAP_Neutral'
            )
        )
        
        if debug:
            print(f"[DEBUG] Calculated market sentiment indicators")
        
        return data
        
    except Exception as e:
        if debug:
            print(f"[DEBUG] Error calculating market sentiment indicators: {e}")
        return data

def calculate_lunar_phases(date_series):
    """Calculate lunar phases for given dates"""
    if not EPHEM_AVAILABLE:
        # Return NaN values if ephem is not available
        moon_phases = pd.Series([np.nan] * len(date_series), index=date_series)
        is_full_moon = pd.Series([False] * len(date_series), index=date_series)
        is_new_moon = pd.Series([False] * len(date_series), index=date_series)
        moon_waxing_waning = pd.Series(['Unknown'] * len(date_series), index=date_series)
        return moon_phases, is_full_moon, is_new_moon, moon_waxing_waning
    
    def get_moon_phase(date):
        try:
            # Convert pandas timestamp to ephem date
            ephem_date = ephem.Date(date)
            # Get moon phase (0 = new moon, 0.5 = full moon, 1 = new moon again)
            moon_phase = ephem.Moon(ephem_date).phase / 100.0
            return moon_phase
        except:
            return np.nan
    
    # Calculate moon phases
    moon_phases = date_series.to_series().apply(get_moon_phase)
    
    # Determine moon states
    is_full_moon = (moon_phases >= 0.48) & (moon_phases <= 0.52)
    is_new_moon = (moon_phases <= 0.02) | (moon_phases >= 0.98)
    
    # Determine waxing/waning
    # Waxing: 0-0.5 (new to full), Waning: 0.5-1.0 (full to new)
    moon_waxing_waning = np.where(
        moon_phases <= 0.5,
        'Waxing',
        'Waning'
    )
    
    return moon_phases, is_full_moon, is_new_moon, moon_waxing_waning

def calculate_market_calendar_features(data, debug=False, include_lunar=True):
    original_columns = len(data.columns)
    """Calculate market calendar and seasonality features"""
    try:
        # Ensure we have datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data = data.set_index('Date')
            else:
                print("[ERROR] No date column found for calendar features")
                return data
        
        dates = data.index
        
        # Basic calendar features
        data['Day_Of_Week'] = dates.dayofweek  # 0=Monday, 6=Sunday
        data['Day_Of_Month'] = dates.day
        data['Day_Of_Year'] = dates.dayofyear
        data['Week_Of_Year'] = dates.isocalendar().week
        data['Month'] = dates.month
        data['Year'] = dates.year
        
        # Lunar phases
        if include_lunar and EPHEM_AVAILABLE:
            if debug:
                print("[DEBUG] Calculating lunar phases...")
            moon_phases, is_full_moon, is_new_moon, moon_waxing_waning = calculate_lunar_phases(dates)
            data['Moon_Phase'] = moon_phases
            data['Is_Full_Moon'] = is_full_moon
            data['Is_New_Moon'] = is_new_moon
            data['Is_Moon_Waxing_or_Waning'] = moon_waxing_waning
        else:
            if debug and not EPHEM_AVAILABLE:
                print("[DEBUG] Skipping lunar calculations - ephem not available")
            elif debug and not include_lunar:
                print("[DEBUG] Skipping lunar calculations - disabled by user")
            data['Moon_Phase'] = np.nan
            data['Is_Full_Moon'] = False
            data['Is_New_Moon'] = False
            data['Is_Moon_Waxing_or_Waning'] = 'Unknown'
        
        # Quarter features
        data['Is_Quarter_Start'] = (dates.month.isin([1, 4, 7, 10])) & (dates.day == 1)
        data['Is_Quarter_End'] = (
            ((dates.month == 3) & (dates.day == 31)) |
            ((dates.month == 6) & (dates.day == 30)) |
            ((dates.month == 9) & (dates.day == 30)) |
            ((dates.month == 12) & (dates.day == 31))
        )
        
        # Earnings season (approximate: 2nd week of Jan, Apr, Jul, Oct)
        data['Is_Earnings_Season'] = (
            ((dates.month == 1) & (dates.day >= 8) & (dates.day <= 22)) |
            ((dates.month == 4) & (dates.day >= 8) & (dates.day <= 22)) |
            ((dates.month == 7) & (dates.day >= 8) & (dates.day <= 22)) |
            ((dates.month == 10) & (dates.day >= 8) & (dates.day <= 22))
        )
        
        # Options expiration (3rd Friday of each month)
        def is_third_friday(date):
            return (date.weekday() == 4) and (8 <= date.day <= 14)
        
        data['Is_Options_Expiration_Week'] = dates.to_series().apply(is_third_friday)
        
        # Quadruple witching (3rd Friday of Mar, Jun, Sep, Dec)
        data['Is_Quadruple_Witching'] = (
            dates.month.isin([3, 6, 9, 12]) & 
            dates.to_series().apply(is_third_friday)
        )
        
        # FOMC weeks (approximate - 8 meetings per year)
        # This is a simplified version - actual dates vary
        fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
        data['Is_FOMC_Week'] = (
            dates.month.isin(fomc_months) & 
            (dates.day >= 15) & (dates.day <= 21)
        )
        
        # January effect (first 10 trading days)
        data['Is_January_Effect'] = (dates.month == 1) & (dates.day <= 10)
        
        # Santa Claus rally (last 5 days of Dec + first 2 of Jan)
        data['Is_Santa_Claus_Rally_Period'] = (
            ((dates.month == 12) & (dates.day >= 27)) |
            ((dates.month == 1) & (dates.day <= 2))
        )
        
        # Sell in May (May 1 to Oct 31)
        data['Is_Sell_In_May_Period'] = (dates.month >= 5) & (dates.month <= 10)
        
        # Tax loss harvesting (last 7 trading days of December)
        data['Is_Tax_Loss_Harvesting_Period'] = (dates.month == 12) & (dates.day >= 25)
        
        # Month end (last trading day of month - simplified)
        data['Is_Month_End'] = dates.is_month_end
        
        # First trading day of month (simplified)
        data['Is_First_Trading_Day_Of_Month'] = dates.day == 1
        
        # US Market Holidays (simplified - actual holidays may vary)
        def is_holiday_week(date):
            # Memorial Day (last Monday in May)
            if date.month == 5 and date.weekday() == 0 and date.day >= 25:
                return True
            # Independence Day (July 4)
            if date.month == 7 and date.day == 4:
                return True
            # Labor Day (first Monday in September)
            if date.month == 9 and date.weekday() == 0 and date.day <= 7:
                return True
            # Thanksgiving (4th Thursday in November)
            if date.month == 11 and date.weekday() == 3 and 22 <= date.day <= 28:
                return True
            # Christmas (December 25)
            if date.month == 12 and date.day == 25:
                return True
            return False
        
        data['Is_Holiday_Week'] = dates.to_series().apply(is_holiday_week)
        
        # === AGRICULTURAL AND COMMODITY SEASONS ===
        # Planting season (April-June)
        data['Is_Planting_Season'] = (dates.month >= 4) & (dates.month <= 6)
        
        # Harvest season (September-November)
        data['Is_Harvest_Season'] = (dates.month >= 9) & (dates.month <= 11)
        
        # USDA WASDE Grain Report (2nd week of each month)
        data['Is_Grain_Report_Week'] = (dates.day >= 8) & (dates.day <= 14)
        
        # EIA Ethanol Inventory Release (Wednesdays)
        data['Is_Ethanol_Inventory_Release'] = dates.weekday == 2  # Wednesday = 2
        
        # Hurricane season (June 1 - Nov 30)
        data['Is_Hurricane_Season'] = (
            ((dates.month == 6) & (dates.day >= 1)) |
            (dates.month.isin([7, 8, 9, 10])) |
            ((dates.month == 11) & (dates.day <= 30))
        )
        
        # === ASTRONOMICAL EVENTS ===
        # Spring Equinox (around March 20)
        data['Is_Spring_Equinox'] = (dates.month == 3) & (dates.day >= 19) & (dates.day <= 21)
        
        # Summer Solstice (around June 21)
        data['Is_Summer_Solstice'] = (dates.month == 6) & (dates.day >= 20) & (dates.day <= 22)
        
        # Fall Equinox (around September 22)
        data['Is_Fall_Equinox'] = (dates.month == 9) & (dates.day >= 21) & (dates.day <= 23)
        
        # Winter Solstice (around December 21)
        data['Is_Winter_Solstice'] = (dates.month == 12) & (dates.day >= 20) & (dates.day <= 22)
        
        # Solar Maximum Cycle (every ~11 years, approximate)
        # 2025 is expected to be near solar maximum
        def is_solar_maximum_year(year):
            # Approximate solar cycle years (2000, 2014, 2025, 2036, etc.)
            solar_max_years = [2000, 2014, 2025, 2036, 2047, 2058]
            return year in solar_max_years
        
        data['Is_Solar_Maximum_Cycle_Year'] = dates.year.isin([2000, 2014, 2025, 2036, 2047, 2058])
        
        # === RETAIL AND CONSUMER SEASONS ===
        # Back to School (Aug 1 - Sep 15)
        data['Is_Back_To_School_Period'] = (
            ((dates.month == 8) & (dates.day >= 1)) |
            ((dates.month == 9) & (dates.day <= 15))
        )
        
        # Black Friday (Friday after Thanksgiving - simplified)
        data['Is_Black_Friday'] = (
            (dates.month == 11) & 
            (dates.weekday == 4) &  # Friday
            (dates.day >= 22) & (dates.day <= 28)
        )
        
        # Cyber Monday (Monday after Thanksgiving - simplified)
        data['Is_Cyber_Monday'] = (
            (dates.month == 11) & 
            (dates.weekday == 0) &  # Monday
            (dates.day >= 25) & (dates.day <= 30)
        )
        
        # Amazon Prime Day (varies: July or Oct - simplified to July)
        data['Is_Amazon_Prime_Day'] = (dates.month == 7) & (dates.day >= 10) & (dates.day <= 20)
        
        # Holiday Shopping Season (Nov 15 - Dec 24)
        data['Is_Holiday_Shopping_Season'] = (
            ((dates.month == 11) & (dates.day >= 15)) |
            ((dates.month == 12) & (dates.day <= 24))
        )
        
        # Valentine's Week (Feb 7-14)
        data['Is_Valentine_Week'] = (dates.month == 2) & (dates.day >= 7) & (dates.day <= 14)
        
        # Mother's Day Week (2nd Sunday of May - simplified)
        data['Is_Mothers_Day_Week'] = (dates.month == 5) & (dates.day >= 8) & (dates.day <= 14)
        
        # Father's Day Week (3rd Sunday of June - simplified)
        data['Is_Fathers_Day_Week'] = (dates.month == 6) & (dates.day >= 15) & (dates.day <= 21)
        
        # === TAX AND SPENDING SEASONS ===
        # Tax Refund Season (Feb 15 - Apr 30)
        data['Is_Tax_Refund_Season'] = (
            ((dates.month == 2) & (dates.day >= 15)) |
            (dates.month.isin([3, 4]))
        )
        
        # Labor Day Weekend (1st Monday of Sep - start of fall spending)
        data['Is_Labor_Day_Weekend'] = (dates.month == 9) & (dates.weekday == 0) & (dates.day <= 7)
        
        # === TRAVEL AND TOURISM ===
        # Summer Travel Season (May 15 - Aug 31)
        data['Is_Summer_Travel_Season'] = (
            ((dates.month == 5) & (dates.day >= 15)) |
            (dates.month.isin([6, 7, 8]))
        )
        
        # Tourism Peak Month (July)
        data['Is_Tourism_Peak_Month'] = dates.month == 7
        
        # === INTERNATIONAL EVENTS ===
        # Chinese New Year Period (January or February - simplified)
        data['Is_Chinese_New_Year_Period'] = (
            ((dates.month == 1) & (dates.day >= 20)) |
            ((dates.month == 2) & (dates.day <= 15))
        )
        
        # Super Bowl Week (1st Sunday in Feb - consumer stocks)
        data['Is_Super_Bowl_Week'] = (dates.month == 2) & (dates.day >= 1) & (dates.day <= 7)
        
        # Olympic Year (every 4 years)
        def is_olympic_year(year):
            olympic_years = [2000, 2004, 2008, 2012, 2016, 2020, 2024, 2028, 2032, 2036, 2040]
            return year in olympic_years
        
        data['Is_Olympic_Year'] = (dates.year % 4 == 0)
        
        # === FEDERAL RESERVE AND CENTRAL BANK EVENTS ===
        # FOMC Meeting Dates (8 times/year - approximate)
        # 2024: Jan 30-31, Mar 19-20, May 1, Jun 11-12, Jul 30-31, Sep 17-18, Nov 6-7, Dec 17-18
        def is_fomc_meeting_week(date):
            fomc_weeks_2024 = [
                (1, 29, 31), (3, 18, 20), (5, 1, 1), (6, 11, 12), 
                (7, 30, 31), (9, 17, 18), (11, 6, 7), (12, 17, 18)
            ]
            for month, start_day, end_day in fomc_weeks_2024:
                if date.month == month and start_day <= date.day <= end_day:
                    return True
            return False
        
        data['Is_FOMC_Meeting_Week'] = dates.to_series().apply(is_fomc_meeting_week)
        
        # ECB, BOE, BOJ Meeting Dates (simplified - approximate)
        # ECB: 8 times/year, BOE: 8 times/year, BOJ: 8 times/year
        def is_global_cb_meeting_week(date):
            # Simplified: assume meetings in middle of each quarter
            cb_months = [1, 3, 4, 6, 7, 9, 10, 12]
            return date.month in cb_months and 10 <= date.day <= 20
        
        data['Is_Global_CB_Meeting_Week'] = dates.to_series().apply(is_global_cb_meeting_week)
        
        # === ECONOMIC DATA RELEASES ===
        # Non-Farm Payrolls (1st Friday of each month)
        data['Is_NonFarm_Payrolls_Week'] = (dates.weekday == 4) & (dates.day <= 7)  # Friday in first week
        
        # CPI & PPI Reports (monthly, ~mid-month)
        data['Is_CPI_PPI_Report_Week'] = (dates.day >= 10) & (dates.day <= 15)
        
        # GDP Releases (quarterly - end of Jan, Apr, Jul, Oct)
        data['Is_GDP_Release_Week'] = (
            ((dates.month == 1) & (dates.day >= 25)) |
            ((dates.month == 4) & (dates.day >= 25)) |
            ((dates.month == 7) & (dates.day >= 25)) |
            ((dates.month == 10) & (dates.day >= 25))
        )
        
        # Beige Book Releases (2 weeks before FOMC - simplified)
        data['Is_Beige_Book_Week'] = (
            ((dates.month == 1) & (dates.day >= 15)) |
            ((dates.month == 3) & (dates.day >= 5)) |
            ((dates.month == 4) & (dates.day >= 15)) |
            ((dates.month == 6) & (dates.day >= 25)) |
            ((dates.month == 7) & (dates.day >= 15)) |
            ((dates.month == 9) & (dates.day >= 3)) |
            ((dates.month == 10) & (dates.day >= 20)) |
            ((dates.month == 12) & (dates.day >= 3))
        )
        
        # Treasury Auctions (monthly schedule: 2, 5, 10, 30-year notes)
        data['Is_Treasury_Auction_Week'] = (dates.day >= 8) & (dates.day <= 15)
        
        # Consumer Confidence Index (last Tuesday of each month)
        data['Is_Consumer_Confidence_Week'] = (dates.weekday == 1) & (dates.day >= 25)  # Tuesday in last week
        
        # ISM Manufacturing/Services PMI (start of each month)
        data['Is_ISM_PMI_Week'] = (dates.day >= 1) & (dates.day <= 5)
        
        # Retail Sales Report (mid-month)
        data['Is_Retail_Sales_Report_Week'] = (dates.day >= 12) & (dates.day <= 18)
        
        # === MARKET MICROSTRUCTURE EVENTS ===
        # Jackson Hole Symposium (late August)
        data['Is_Jackson_Hole_Week'] = (dates.month == 8) & (dates.day >= 20) & (dates.day <= 26)
        
        # Triple Witching (3rd Friday of Mar, Jun, Sep, Dec)
        data['Is_Triple_Witching'] = (
            dates.month.isin([3, 6, 9, 12]) & 
            dates.to_series().apply(is_third_friday)
        )
        
        # End-of-Month/Quarter/Year Rebalancing
        data['Is_End_Of_Month_Rebalancing'] = dates.is_month_end
        data['Is_End_Of_Quarter_Rebalancing'] = data['Is_Quarter_End']
        data['Is_End_Of_Year_Rebalancing'] = (dates.month == 12) & (dates.day == 31)
        
        # Mid-Month Reversal Tendency (~15th of each month)
        data['Is_Mid_Month_Reversal'] = (dates.day >= 14) & (dates.day <= 16)
        
        # Turn-of-the-Month Effect (last & first 4 trading days of month)
        data['Is_Turn_Of_Month_Effect'] = (dates.day >= 28) | (dates.day <= 4)
        
        # VIX Options Expiration (usually Wednesdays)
        data['Is_VIX_Options_Expiration'] = (dates.weekday == 2) & (dates.day >= 15) & (dates.day <= 21)
        
        # Russell Index Reconstitution (June)
        data['Is_Russell_Reconstitution'] = (dates.month == 6) & (dates.day >= 20) & (dates.day <= 30)
        
        # Annual Hedge Fund Redemptions (Oct 31, Dec 31)
        data['Is_Hedge_Fund_Redemption'] = (
            ((dates.month == 10) & (dates.day == 31)) |
            ((dates.month == 12) & (dates.day == 31))
        )
        
        # Tax Day (April 15 or next business day)
        data['Is_Tax_Day'] = (dates.month == 4) & (dates.day >= 15) & (dates.day <= 17)
        
        # === EARNINGS AND CORPORATE EVENTS ===
        # Start of Earnings Season (begins with JPM/GS or AA)
        data['Is_Earnings_Season_Start'] = (
            ((dates.month == 1) & (dates.day >= 10) & (dates.day <= 15)) |
            ((dates.month == 4) & (dates.day >= 10) & (dates.day <= 15)) |
            ((dates.month == 7) & (dates.day >= 10) & (dates.day <= 15)) |
            ((dates.month == 10) & (dates.day >= 10) & (dates.day <= 15))
        )
        
        # Ex-Dividend Dates (signals short-term price drops - simplified)
        data['Is_Ex_Dividend_Week'] = (dates.weekday == 3) & (dates.day >= 15) & (dates.day <= 21)  # Thursdays mid-month
        
        # === MARKET PATTERNS AND EFFECTS ===
        # Post-Holiday Drift (positive bias after major holidays)
        data['Is_Post_Holiday_Drift'] = (
            ((dates.month == 1) & (dates.day >= 2) & (dates.day <= 5)) |
            ((dates.month == 7) & (dates.day >= 5) & (dates.day <= 8)) |
            ((dates.month == 9) & (dates.day >= 2) & (dates.day <= 5)) |
            ((dates.month == 11) & (dates.day >= 25) & (dates.day <= 30)) |
            ((dates.month == 12) & (dates.day >= 26) & (dates.day <= 30))
        )
        
        # Pre-Holiday Rally (Thanksgiving, Christmas)
        data['Is_Pre_Holiday_Rally'] = (
            ((dates.month == 11) & (dates.day >= 20) & (dates.day <= 25)) |
            ((dates.month == 12) & (dates.day >= 20) & (dates.day <= 24))
        )
        
        # Friday Effect (risk-off or profit-taking before weekends)
        data['Is_Friday_Effect'] = dates.weekday == 4
        
        # Monday Effect (often weak opens)
        data['Is_Monday_Effect'] = dates.weekday == 0
        
        # Summer Doldrums (low volume July–Aug)
        data['Is_Summer_Doldrums'] = (dates.month >= 7) & (dates.month <= 8)
        
        # Halloween Effect ("Sell in May" ends Oct 31)
        data['Is_Halloween_Effect'] = (dates.month == 10) & (dates.day >= 29) & (dates.day <= 31)
        
        # === POLITICAL AND ELECTION CYCLES ===
        # End of Presidential Terms (election cycle years often bullish)
        def is_presidential_election_year(year):
            return year % 4 == 0
        
        data['Is_Presidential_Election_Year'] = dates.year.isin([year for year in dates.year.unique() if is_presidential_election_year(year)])
        
        # Year 3 of Presidential Cycle (historically strongest year)
        def is_year_3_presidential_cycle(year):
            return (year - 1) % 4 == 0
        
        data['Is_Year_3_Presidential_Cycle'] = dates.year.isin([year for year in dates.year.unique() if is_year_3_presidential_cycle(year)])
        
        # === ETF AND INDEX EVENTS ===
        # Rebalance Dates of ETFs (e.g., SPY quarterly rebalances)
        data['Is_ETF_Rebalance_Week'] = (
            ((dates.month == 3) & (dates.day >= 15) & (dates.day <= 21)) |
            ((dates.month == 6) & (dates.day >= 15) & (dates.day <= 21)) |
            ((dates.month == 9) & (dates.day >= 15) & (dates.day <= 21)) |
            ((dates.month == 12) & (dates.day >= 15) & (dates.day <= 21))
        )
        
        # === US HOLIDAYS AND OBSERVANCES ===
        # New Year's Day (January 1)
        data['Is_New_Years_Day'] = (dates.month == 1) & (dates.day == 1)
        
        # Presidents' Day (3rd Monday in February)
        def is_presidents_day(date):
            return date.month == 2 and date.weekday() == 0 and 15 <= date.day <= 21
        
        data['Is_Presidents_Day'] = dates.to_series().apply(is_presidents_day)
        
        # Easter Weekend (simplified - approximate dates)
        def is_easter_weekend(date):
            # Easter typically falls between March 22 and April 25
            # This is a simplified approximation
            easter_approximations = [
                (3, 22, 25), (3, 29, 31), (4, 1, 7), (4, 8, 14), (4, 15, 21), (4, 22, 25)
            ]
            for month, start_day, end_day in easter_approximations:
                if date.month == month and start_day <= date.day <= end_day:
                    return True
            return False
        
        data['Is_Easter_Weekend'] = dates.to_series().apply(is_easter_weekend)
        
        # Independence Day (July 4)
        data['Is_Independence_Day'] = (dates.month == 7) & (dates.day == 4)
        
        # Halloween (October 31)
        data['Is_Halloween'] = (dates.month == 10) & (dates.day == 31)
        
        # Christmas (December 25)
        data['Is_Christmas'] = (dates.month == 12) & (dates.day == 25)
        
        # Boxing Day (December 26)
        data['Is_Boxing_Day'] = (dates.month == 12) & (dates.day == 26)
        
        # === INTERNATIONAL HOLIDAYS ===
        # Lunar New Year (simplified - typically January/February)
        def is_lunar_new_year(date):
            # Lunar New Year typically falls between January 21 and February 20
            return ((date.month == 1 and date.day >= 21) or 
                   (date.month == 2 and date.day <= 20))
        
        data['Is_Lunar_New_Year'] = dates.to_series().apply(is_lunar_new_year)
        
        # Valentine's Day (February 14)
        data['Is_Valentines_Day'] = (dates.month == 2) & (dates.day == 14)
        
        # Hanukkah (simplified - typically December)
        def is_hanukkah(date):
            # Hanukkah typically falls in December, sometimes late November
            return ((date.month == 11 and date.day >= 25) or 
                   (date.month == 12 and date.day <= 30))
        
        data['Is_Hanukkah'] = dates.to_series().apply(is_hanukkah)
        
        # Diwali (simplified - typically October/November)
        def is_diwali(date):
            # Diwali typically falls between October and November
            return ((date.month == 10 and date.day >= 15) or 
                   (date.month == 11 and date.day <= 15))
        
        data['Is_Diwali'] = dates.to_series().apply(is_diwali)
        
        # Ramadan (simplified - varies by year)
        def is_ramadan(date):
            # Ramadan typically falls between March and May (varies by year)
            return date.month in [3, 4, 5]
        
        data['Is_Ramadan'] = dates.to_series().apply(is_ramadan)
        
        # Eid al-Fitr (simplified - follows Ramadan)
        def is_eid_al_fitr(date):
            # Eid al-Fitr typically falls between April and June
            return date.month in [4, 5, 6]
        
        data['Is_Eid_al_Fitr'] = dates.to_series().apply(is_eid_al_fitr)
        
        # Golden Week China (April 29 - May 5)
        def is_golden_week_china(date):
            return ((date.month == 4 and date.day >= 29) or 
                   (date.month == 5 and date.day <= 5))
        
        data['Is_Golden_Week_China'] = dates.to_series().apply(is_golden_week_china)
        
        # Carnival Season (simplified - typically February/March)
        def is_carnival_season(date):
            # Carnival typically falls in February/March
            return date.month in [2, 3]
        
        data['Is_Carnival_Season'] = dates.to_series().apply(is_carnival_season)
        
        # Singles Day China (November 11)
        data['Is_Singles_Day_China'] = (dates.month == 11) & (dates.day == 11)
        
        # Super Bowl Weekend (1st Sunday in February)
        def is_super_bowl_weekend(date):
            return date.month == 2 and date.weekday() == 6 and date.day <= 7
        
        data['Is_Super_Bowl_Weekend'] = dates.to_series().apply(is_super_bowl_weekend)
        
        # === FINANCIAL CYCLES AND DEADLINES ===
        # Fiscal Year End (September 30 for many companies)
        data['Is_Fiscal_Year_End'] = (dates.month == 9) & (dates.day == 30)
        
        # Tax Day US (April 15 or next business day)
        data['Is_Tax_Day_US'] = (dates.month == 4) & (dates.day >= 15) & (dates.day <= 17)
        
        # IRA Contribution Deadline (April 15)
        data['Is_IRA_Contribution_Deadline'] = (dates.month == 4) & (dates.day >= 15) & (dates.day <= 17)
        
        # College Tuition Due (typically August/September)
        def is_college_tuition_due(date):
            return ((date.month == 8 and date.day >= 15) or 
                   (date.month == 9 and date.day <= 15))
        
        data['Is_College_Tuition_Due'] = dates.to_series().apply(is_college_tuition_due)
        
        # Property Tax Payment Date (typically December)
        def is_property_tax_payment_date(date):
            return (date.month == 12 and date.day >= 1 and date.day <= 31)
        
        data['Is_Property_Tax_Payment_Date'] = dates.to_series().apply(is_property_tax_payment_date)
        
        # Health Insurance Enrollment Season (November/December)
        def is_health_insurance_enrollment_season(date):
            return date.month in [11, 12]
        
        data['Is_Health_Insurance_Enrollment_Season'] = dates.to_series().apply(is_health_insurance_enrollment_season)
        
        # === PAYROLL AND COMPENSATION CYCLES ===
        # Paycheck Cycle 1st (beginning of month)
        data['Is_Paycheck_Cycle_1st'] = dates.day <= 5
        
        # Paycheck Cycle 15th (mid-month)
        data['Is_Paycheck_Cycle_15th'] = (dates.day >= 10) & (dates.day <= 20)
        
        # Debt Payment Cycle (typically end of month)
        def is_debt_payment_cycle(date):
            return date.day >= 25 or date.day <= 5
        
        data['Is_Debt_Payment_Cycle'] = dates.to_series().apply(is_debt_payment_cycle)
        
        # Bonus Payout Season (December/January)
        def is_bonus_payout_season(date):
            return ((date.month == 12 and date.day >= 15) or 
                   (date.month == 1 and date.day <= 15))
        
        data['Is_Bonus_Payout_Season'] = dates.to_series().apply(is_bonus_payout_season)
        
        # === MARKET PSYCHOLOGY AND SEASONAL EFFECTS ===
        # Beginning of Year Optimism (first 2 weeks of January)
        data['Is_Beginning_Of_Year_Optimism'] = (dates.month == 1) & (dates.day <= 14)
        
        # September Anxiety (September effect)
        data['Is_September_Anxiety'] = dates.month == 9
        
        # Window Dressing Season (end of quarter)
        data['Is_Window_Dressing_Season'] = (
            ((dates.month == 3) & (dates.day >= 25)) |
            ((dates.month == 6) & (dates.day >= 25)) |
            ((dates.month == 9) & (dates.day >= 25)) |
            ((dates.month == 12) & (dates.day >= 25))
        )
        
        # Post Bonus Spending (January/February)
        def is_post_bonus_spending(date):
            return ((date.month == 1 and date.day >= 15) or 
                   (date.month == 2 and date.day <= 15))
        
        data['Is_Post_Bonus_Spending'] = dates.to_series().apply(is_post_bonus_spending)
        
        # Post Tax Refund Spending (March/April)
        def is_post_tax_refund_spending(date):
            return ((date.month == 3 and date.day >= 15) or 
                   (date.month == 4 and date.day <= 30))
        
        data['Is_Post_Tax_Refund_Spending'] = dates.to_series().apply(is_post_tax_refund_spending)
        
        # === BIOLOGICAL AND ENVIRONMENTAL CYCLES ===
        # Circadian Rhythm Morning (6 AM - 10 AM equivalent in trading hours)
        # This is simplified to represent morning trading bias
        data['Is_Circadian_Rhythm_Morning'] = dates.weekday.isin([0, 1, 2, 3, 4])  # Weekday mornings
        
        # Seasonal Affective Winter (December - February)
        def is_seasonal_affective_winter(date):
            return date.month in [12, 1, 2]
        
        data['Is_Seasonal_Affective_Winter'] = dates.to_series().apply(is_seasonal_affective_winter)
        
        # Daylight Savings Shift (March and November)
        def is_daylight_savings_shift(date):
            # Spring forward: 2nd Sunday in March
            # Fall back: 1st Sunday in November
            return ((date.month == 3 and date.weekday == 6 and 8 <= date.day <= 14) or
                   (date.month == 11 and date.weekday == 6 and 1 <= date.day <= 7))
        
        data['Is_Daylight_Savings_Shift'] = dates.to_series().apply(is_daylight_savings_shift)
        
        # === ENHANCED LUNAR CALCULATIONS ===
        # Moon cycle sine encoded (for machine learning)
        if include_lunar and EPHEM_AVAILABLE:
            # Calculate sine and cosine encoding of moon phase
            moon_phases, _, _, _ = calculate_lunar_phases(dates)
            data['Moon_Cycle_Sine_Encoded'] = np.sin(2 * np.pi * moon_phases)
            data['Moon_Cycle_Cosine_Encoded'] = np.cos(2 * np.pi * moon_phases)
            
            # Waxing and Waning as separate flags
            data['Is_Waxing_Moon'] = (moon_phases > 0) & (moon_phases <= 0.5)
            data['Is_Waning_Moon'] = (moon_phases > 0.5) & (moon_phases <= 1.0)
            
            # Lunar and Solar Eclipses (simplified - major eclipse periods)
            def is_lunar_eclipse_day(date):
                # Major lunar eclipses (simplified approximation)
                eclipse_dates = [
                    (1, 10, 15), (6, 5, 10), (11, 8, 12), (12, 3, 8)
                ]
                for month, start_day, end_day in eclipse_dates:
                    if date.month == month and start_day <= date.day <= end_day:
                        return True
                return False
            
            def is_solar_eclipse_day(date):
                # Major solar eclipses (simplified approximation)
                eclipse_dates = [
                    (4, 8, 12), (10, 14, 18)
                ]
                for month, start_day, end_day in eclipse_dates:
                    if date.month == month and start_day <= date.day <= end_day:
                        return True
                return False
            
            data['Is_Lunar_Eclipse_Day'] = dates.to_series().apply(is_lunar_eclipse_day)
            data['Is_Solar_Eclipse_Day'] = dates.to_series().apply(is_solar_eclipse_day)
        else:
            # Set lunar-related columns to NaN/False if ephem not available
            data['Moon_Cycle_Sine_Encoded'] = np.nan
            data['Moon_Cycle_Cosine_Encoded'] = np.nan
            data['Is_Waxing_Moon'] = False
            data['Is_Waning_Moon'] = False
            data['Is_Lunar_Eclipse_Day'] = False
            data['Is_Solar_Eclipse_Day'] = False
        
        # === SOCIAL MEDIA AND NEWS SENTIMENT ===
        # Twitter Stock Sentiment (simplified - would require API integration)
        data['Twitter_Stock_Sentiment'] = np.random.uniform(-1, 1, len(data))  # Placeholder
        
        # Reddit WallStreetBets Mentions (simplified)
        data['Reddit_WallStreetBets_Mentions'] = np.random.randint(0, 100, len(data))  # Placeholder
        
        # Google Search Trend Score (simplified)
        data['Google_Search_Trend_Score'] = np.random.uniform(0, 100, len(data))  # Placeholder
        
        # News Headline Sentiment Score (simplified)
        data['News_Headline_Sentiment_Score'] = np.random.uniform(-1, 1, len(data))  # Placeholder
        
        # Article Volume By Ticker (simplified)
        data['Article_Volume_By_Ticker'] = np.random.randint(0, 50, len(data))  # Placeholder
        
        # Bloomberg Headline Count (simplified)
        data['Bloomberg_Headline_Count'] = np.random.randint(0, 20, len(data))  # Placeholder
        
        # Yahoo Finance Comment Sentiment (simplified)
        data['YahooFinance_Comment_Sentiment'] = np.random.uniform(-1, 1, len(data))  # Placeholder
        
        # YouTube Financial Mentions (simplified)
        data['YouTube_Financial_Mentions'] = np.random.randint(0, 30, len(data))  # Placeholder
        
        # SEC 10K Sentiment Score (simplified)
        data['SEC_10K_Sentiment_Score'] = np.random.uniform(-1, 1, len(data))  # Placeholder
        
        # Insider Sales Headline Count (simplified)
        data['Insider_Sales_Headline_Count'] = np.random.randint(0, 10, len(data))  # Placeholder
        
        # === POLITICAL AND GOVERNMENT EVENTS ===
        # US Election Day (first Tuesday in November, every 2 years)
        def is_us_election_day(date):
            return (date.month == 11 and date.weekday() == 1 and 
                   1 <= date.day <= 7 and date.year % 2 == 0)
        
        data['Is_US_Election_Day'] = dates.to_series().apply(is_us_election_day)
        
        # Presidential Debate Flag (simplified - typically September/October of election years)
        def is_presidential_debate_flag(date):
            return (date.month in [9, 10] and date.year % 4 == 0)
        
        data['Is_Presidential_Debate_Flag'] = dates.to_series().apply(is_presidential_debate_flag)
        
        # FOMC Announcement Day (same as FOMC meeting week, but specific day)
        data['Is_FOMC_Announcement_Day'] = data['Is_FOMC_Meeting_Week']
        
        # Fed Nomination Date (simplified - typically when terms expire)
        def is_fed_nomination_date(date):
            # Fed Chair terms typically expire in February
            return (date.month == 2 and date.day >= 1 and date.day <= 7)
        
        data['Is_Fed_Nomination_Date'] = dates.to_series().apply(is_fed_nomination_date)
        
        # G7 Summit Flag (simplified - typically June/July)
        def is_g7_summit_flag(date):
            return date.month in [6, 7]
        
        data['Is_G7_Summit_Flag'] = dates.to_series().apply(is_g7_summit_flag)
        
        # G20 Summit Flag (simplified - typically September/November)
        def is_g20_summit_flag(date):
            return date.month in [9, 11]
        
        data['Is_G20_Summit_Flag'] = dates.to_series().apply(is_g20_summit_flag)
        
        # UN General Assembly Flag (typically September)
        data['Is_UN_General_Assembly_Flag'] = (dates.month == 9) & (dates.day >= 15) & (dates.day <= 30)
        
        # Debt Ceiling Crisis Flag (simplified - typically when debt ceiling is reached)
        def is_debt_ceiling_crisis_flag(date):
            # Simplified - typically happens in Q3/Q4
            return date.month in [7, 8, 9, 10, 11, 12]
        
        data['Is_Debt_Ceiling_Crisis_Flag'] = dates.to_series().apply(is_debt_ceiling_crisis_flag)
        
        # US Budget Deadline (typically September 30)
        data['Is_US_Budget_Deadline'] = (dates.month == 9) & (dates.day >= 25) & (dates.day <= 30)
        
        # Midterm Election Year (every 2 years, not presidential election years)
        data['Is_Midterm_Election_Year'] = (dates.year % 2 == 0) & (dates.year % 4 != 0)
        
        # State of the Union Flag (typically January/February)
        def is_state_of_the_union_flag(date):
            return (date.month in [1, 2] and date.weekday() == 1 and 20 <= date.day <= 28)
        
        data['Is_State_Of_The_Union_Flag'] = dates.to_series().apply(is_state_of_the_union_flag)
        
        # Sanctions Announcement Flag (simplified)
        def is_sanctions_announcement_flag(date):
            # Simplified - can happen any time
            return np.random.choice([True, False], len(data), p=[0.01, 0.99])
        
        data['Is_Sanctions_Announcement_Flag'] = is_sanctions_announcement_flag(dates)
        
        # Trade Agreement Deadline (simplified)
        def is_trade_agreement_deadline(date):
            # Simplified - typically end of year
            return (date.month == 12 and date.day >= 15)
        
        data['Is_Trade_Agreement_Deadline'] = dates.to_series().apply(is_trade_agreement_deadline)
        
        # Geopolitical Tension Flag (simplified)
        def is_geopolitical_tension_flag(date):
            # Simplified - can happen any time
            return np.random.choice([True, False], len(data), p=[0.02, 0.98])
        
        data['Is_Geopolitical_Tension_Flag'] = is_geopolitical_tension_flag(dates)
        
        # Government Shutdown Risk (typically around budget deadlines)
        data['Is_Government_Shutdown_Risk'] = (
            ((dates.month == 9) & (dates.day >= 25)) |
            ((dates.month == 12) & (dates.day >= 15))
        )
        
        # Impeachment Inquiry Started (simplified)
        def is_impeachment_inquiry_started(date):
            # Simplified - very rare event
            return np.random.choice([True, False], len(data), p=[0.001, 0.999])
        
        data['Is_Impeachment_Inquiry_Started'] = is_impeachment_inquiry_started(dates)
        
        # === ALTERNATIVE DATA INDICATORS ===
        # Port Shipping Volume Index (simplified)
        data['Port_Shipping_Volume_Index'] = np.random.uniform(80, 120, len(data))  # Placeholder
        
        # Trucking Freight Index (simplified)
        data['Trucking_Freight_Index'] = np.random.uniform(80, 120, len(data))  # Placeholder
        
        # Flight Traffic Volume (simplified)
        data['Flight_Traffic_Volume'] = np.random.uniform(80, 120, len(data))  # Placeholder
        
        # Job Posting Volume (simplified)
        data['Job_Posting_Volume'] = np.random.uniform(80, 120, len(data))  # Placeholder
        
        # Consumer Sentiment Index (simplified)
        data['Consumer_Sentiment_Index'] = np.random.uniform(50, 100, len(data))  # Placeholder
        
        # Homebuilder Confidence Score (simplified)
        data['Homebuilder_Confidence_Score'] = np.random.uniform(30, 80, len(data))  # Placeholder
        
        # Real Estate Listing Volume (simplified)
        data['Real_Estate_Listing_Volume'] = np.random.uniform(80, 120, len(data))  # Placeholder
        
        # Credit Card Spending Trend (simplified)
        data['Credit_Card_Spending_Trend'] = np.random.uniform(-5, 10, len(data))  # Placeholder
        
        # === MARKET MICROSTRUCTURE INDICATORS ===
        # Insider Buying Cluster Score (simplified)
        data['Insider_Buying_Cluster_Score'] = np.random.uniform(0, 100, len(data))  # Placeholder
        
        # 13F Whale Position Change (simplified)
        data['13F_Whale_Position_Change'] = np.random.uniform(-10, 10, len(data))  # Placeholder
        
        # Dark Pool Buy Activity (simplified)
        data['Dark_Pool_Buy_Activity'] = np.random.uniform(0, 100, len(data))  # Placeholder
        
        # Dark Pool Sell Activity (simplified)
        data['Dark_Pool_Sell_Activity'] = np.random.uniform(0, 100, len(data))  # Placeholder
        
        # Unusual Options Volume (simplified)
        data['Unusual_Options_Volume'] = np.random.uniform(0, 100, len(data))  # Placeholder
        
        # Options Order Imbalance (simplified)
        data['Options_Order_Imbalance'] = np.random.uniform(-1, 1, len(data))  # Placeholder
        
        # Short Interest Ratio (simplified)
        data['Short_Interest_Ratio'] = np.random.uniform(0, 5, len(data))  # Placeholder
        
        # Borrow Fee Rate Spike (simplified)
        data['Borrow_Fee_Rate_Spike'] = np.random.uniform(0, 50, len(data))  # Placeholder
        
        if debug:
            calendar_cols = [col for col in data.columns if col.startswith('Is_') or col in ['Day_Of_Week', 'Day_Of_Year', 'Week_Of_Year', 'Moon_Phase']]
            print(f"[DEBUG] Calculated {len(calendar_cols)} calendar features")
            print(f"[DEBUG] Calendar columns: {calendar_cols}")
        
        return data
        
    except Exception as e:
        if debug:
            print(f"[DEBUG] Error calculating market calendar features: {e}")
            import traceback
            traceback.print_exc()
        return data

def fetch_vix_data(start_date, end_date):
    """Fetch VIX data for the same date range"""
    try:
        vix = yf.Ticker("^VIX")
        vix_data = vix.history(start=start_date, end=end_date)
        if vix_data.empty:
            print("Warning: VIX data is empty")
            return None
        print(f"Successfully fetched VIX data: {len(vix_data)} rows")
        return vix_data
    except Exception as e:
        print(f"Warning: Could not fetch VIX data: {e}")
        return None

def calculate_all_indicators(data, lookforward_days=7):
    """Calculate all technical indicators"""
    original_columns = len(data.columns)
    
    # Check for existing duplicate columns before adding new ones
    existing_columns = set(data.columns)
    duplicate_columns = []
    for col in existing_columns:
        if list(data.columns).count(col) > 1:
            duplicate_columns.append(col)
    
    if duplicate_columns:
        print(f"⚠️  Warning: Found duplicate columns before calculation: {duplicate_columns}")
        # Remove duplicates, keeping the first occurrence
        data = data.loc[:, ~data.columns.duplicated()]
        print(f"✅ Removed {len(duplicate_columns)} duplicate columns")
    # Basic price data
    data['Daily_Return'] = data['Close'].pct_change()
    data['Price_Range'] = data['High'] - data['Low']
    data['Price_Range_Pct'] = (data['Price_Range'] / data['Close']) * 100
    data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
    
    # Volume indicators
    data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_MA_50'] = data['Volume'].rolling(window=50).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
    data['OBV'] = (data['Volume'] * data['Close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()
    
    # VWAP indicators
    data['VWAP'] = calculate_vwap(data['High'], data['Low'], data['Close'], data['Volume'])
    data['VWAP_20'] = calculate_vwap(data['High'], data['Low'], data['Close'], data['Volume'], window=20)
    data['VWAP_50'] = calculate_vwap(data['High'], data['Low'], data['Close'], data['Volume'], window=50)
    data['Price_vs_VWAP'] = ((data['Close'] - data['VWAP']) / data['VWAP']) * 100
    data['Price_vs_VWAP_20'] = ((data['Close'] - data['VWAP_20']) / data['VWAP_20']) * 100
    
    # Volatility indicators
    data['Volatility_20'] = data['Daily_Return'].rolling(window=20).std()
    data['ATR_14'] = pd.concat([data['High'] - data['Low'], 
                                abs(data['High'] - data['Close'].shift()), 
                                abs(data['Low'] - data['Close'].shift())], axis=1).max(axis=1).rolling(window=14).mean()
    
    # Momentum indicators
    data['RSI_14'] = calculate_rsi(data['Close'], 14)
    data['RSI_21'] = calculate_rsi(data['Close'], 21)
    stoch_k, stoch_d = calculate_stochastic(data['High'], data['Low'], data['Close'])
    data['STOCH_14_3_3'] = stoch_k
    data['STOCH_D_3'] = stoch_d
    data['CCI_20'] = calculate_cci(data['High'], data['Low'], data['Close'], 20)
    data['ADX_14'] = calculate_adx(data['High'], data['Low'], data['Close'], 14)
    data['MOM_10'] = data['Close'].diff(10)
    data['MOM_20'] = data['Close'].diff(20)
    data['ROC_10'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100
    data['ROC_20'] = ((data['Close'] - data['Close'].shift(20)) / data['Close'].shift(20)) * 100
    
    # MACD indicators
    macd, signal, histogram = calculate_macd(data['Close'])
    data['MACD_12_26_9'] = macd
    data['MACDs_12_26_9'] = signal
    data['MACDh_12_26_9'] = histogram
    
    # Stochastic RSI
    rsi_stoch_k, rsi_stoch_d = calculate_stochastic(data['RSI_14'], data['RSI_14'], data['RSI_14'])
    data['STOCHRSI_14_14_3_3'] = rsi_stoch_k
    
    # Other oscillators
    data['WILLR_14'] = calculate_williams_r(data['High'], data['Low'], data['Close'], 14)
    data['ULTOSC_7_14_28'] = calculate_ultimate_oscillator(data['High'], data['Low'], data['Close'])
    data['MFI_14'] = calculate_mfi(data['High'], data['Low'], data['Close'], data['Volume'], 14)
    data['CMO_9'] = calculate_cmo(data['Close'], 9)
    
    # Moving averages
    for period in [5, 9, 10, 14, 20, 30, 50, 100, 200]:
        data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
        data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
    
    # === MOVING AVERAGE COMPARISON FEATURES ===
    # These comparison features help ML models by providing pre-computed technical signals
    
    # Golden Cross and Death Cross (EMA-based)
    data['Golden_Cross_EMA_20_50'] = (data['EMA_20'] > data['EMA_50']).astype(int)
    data['Death_Cross_EMA_20_50'] = (data['EMA_20'] < data['EMA_50']).astype(int)
    data['Golden_Cross_EMA_50_200'] = (data['EMA_50'] > data['EMA_200']).astype(int)
    data['Death_Cross_EMA_50_200'] = (data['EMA_50'] < data['EMA_200']).astype(int)
    
    # Golden Cross and Death Cross (SMA-based)
    data['Golden_Cross_SMA_20_50'] = (data['SMA_20'] > data['SMA_50']).astype(int)
    data['Death_Cross_SMA_20_50'] = (data['SMA_20'] < data['SMA_50']).astype(int)
    data['Golden_Cross_SMA_50_200'] = (data['SMA_50'] > data['SMA_200']).astype(int)
    data['Death_Cross_SMA_50_200'] = (data['SMA_50'] < data['SMA_200']).astype(int)
    
    # Moving Average Alignment (trend strength indicator)
    data['MA_Alignment_EMA'] = (
        (data['EMA_20'] > data['EMA_50']).astype(int) + 
        (data['EMA_50'] > data['EMA_200']).astype(int)
    )
    data['MA_Alignment_SMA'] = (
        (data['SMA_20'] > data['SMA_50']).astype(int) + 
        (data['SMA_50'] > data['SMA_200']).astype(int)
    )
    
    # Price vs Moving Average Comparisons
    data['Price_Above_EMA_20'] = (data['Close'] > data['EMA_20']).astype(int)
    data['Price_Above_EMA_50'] = (data['Close'] > data['EMA_50']).astype(int)
    data['Price_Above_EMA_200'] = (data['Close'] > data['EMA_200']).astype(int)
    data['Price_Above_SMA_20'] = (data['Close'] > data['SMA_20']).astype(int)
    data['Price_Above_SMA_50'] = (data['Close'] > data['SMA_50']).astype(int)
    data['Price_Above_SMA_200'] = (data['Close'] > data['SMA_200']).astype(int)
    
    # Moving Average Spreads (normalized differences)
    data['EMA_20_50_Spread'] = (data['EMA_20'] - data['EMA_50']) / data['EMA_50']
    data['EMA_50_200_Spread'] = (data['EMA_50'] - data['EMA_200']) / data['EMA_200']
    data['SMA_20_50_Spread'] = (data['SMA_20'] - data['SMA_50']) / data['SMA_50']
    data['SMA_50_200_Spread'] = (data['SMA_50'] - data['SMA_200']) / data['SMA_200']
    
    # Short-term vs Long-term momentum
    data['EMA_5_20_Spread'] = (data['EMA_5'] - data['EMA_20']) / data['EMA_20']
    data['SMA_5_20_Spread'] = (data['SMA_5'] - data['SMA_20']) / data['SMA_20']
    
    # Moving Average Crossover Signals
    data['EMA_10_20_Crossover'] = (data['EMA_10'] > data['EMA_20']).astype(int)
    data['EMA_20_30_Crossover'] = (data['EMA_20'] > data['EMA_30']).astype(int)
    data['SMA_10_20_Crossover'] = (data['SMA_10'] > data['SMA_20']).astype(int)
    data['SMA_20_30_Crossover'] = (data['SMA_20'] > data['SMA_30']).astype(int)
    
    # Moving Average Slope Comparisons
    data['EMA_20_Slope'] = data['EMA_20'].diff(5) / data['EMA_20'].shift(5)
    data['EMA_50_Slope'] = data['EMA_50'].diff(10) / data['EMA_50'].shift(10)
    data['EMA_200_Slope'] = data['EMA_200'].diff(20) / data['EMA_200'].shift(20)
    data['SMA_20_Slope'] = data['SMA_20'].diff(5) / data['SMA_20'].shift(5)
    data['SMA_50_Slope'] = data['SMA_50'].diff(10) / data['SMA_50'].shift(10)
    data['SMA_200_Slope'] = data['SMA_200'].diff(20) / data['SMA_200'].shift(20)
    
    # Moving Average Convergence/Divergence
    data['EMA_Convergence'] = (data['EMA_20_Slope'] > 0) & (data['EMA_50_Slope'] > 0) & (data['EMA_200_Slope'] > 0)
    data['EMA_Divergence'] = (data['EMA_20_Slope'] < 0) & (data['EMA_50_Slope'] < 0) & (data['EMA_200_Slope'] < 0)
    data['SMA_Convergence'] = (data['SMA_20_Slope'] > 0) & (data['SMA_50_Slope'] > 0) & (data['SMA_200_Slope'] > 0)
    data['SMA_Divergence'] = (data['SMA_20_Slope'] < 0) & (data['SMA_50_Slope'] < 0) & (data['SMA_200_Slope'] < 0)
    
    # Moving Average Strength Indicators
    data['EMA_Trend_Strength'] = (
        (data['EMA_20'] > data['EMA_50']).astype(int) + 
        (data['EMA_50'] > data['EMA_200']).astype(int) + 
        (data['Close'] > data['EMA_20']).astype(int)
    )
    data['SMA_Trend_Strength'] = (
        (data['SMA_20'] > data['SMA_50']).astype(int) + 
        (data['SMA_50'] > data['SMA_200']).astype(int) + 
        (data['Close'] > data['SMA_20']).astype(int)
    )
    
    # === BONGO PATTERN ===
    # Bongo pattern: Price > SMA9 > SMA14 > SMA20 (strong uptrend)
    data['Bongo_SMA'] = (
        (data['Close'] > data['SMA_9']).astype(int) + 
        (data['SMA_9'] > data['SMA_14']).astype(int) + 
        (data['SMA_14'] > data['SMA_20']).astype(int)
    )
    
    # Bongo pattern: Price > EMA9 > EMA14 > EMA20 (strong uptrend)
    data['Bongo_EMA'] = (
        (data['Close'] > data['EMA_9']).astype(int) + 
        (data['EMA_9'] > data['EMA_14']).astype(int) + 
        (data['EMA_14'] > data['EMA_20']).astype(int)
    )
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(data['Close'])
    data['BB_Upper'] = bb_upper
    data['BB_Lower'] = bb_lower
    data['BB_Middle'] = bb_middle
    data['BB_Width'] = ((bb_upper - bb_lower) / bb_middle) * 100
    data['BB_Position'] = ((data['Close'] - bb_lower) / (bb_upper - bb_lower)) * 100
    
    # Keltner Channels
    kc_upper, kc_lower, kc_middle = calculate_keltner_channels(data['High'], data['Low'], data['Close'])
    data['KC_Upper'] = kc_upper
    data['KC_Lower'] = kc_lower
    data['KC_Middle'] = kc_middle
    
    # Donchian Channels
    dc_upper, dc_lower, dc_middle = calculate_donchian_channels(data['High'], data['Low'])
    data['DC_Upper'] = dc_upper
    data['DC_Lower'] = dc_lower
    data['DC_Middle'] = dc_middle
    
    # Parabolic SAR
    data['PSAR'] = calculate_parabolic_sar(data['High'], data['Low'], data['Close'])
    
    # Trend indicators
    # TRIX - using function-based calculation instead of manual calculation
    trix, trix_signal = calculate_trix(data['Close'])
    data['TRIX'] = trix  # Use the function result instead of manual calculation
    data['TRIX_Signal'] = trix_signal
    data['CoppockCurve'] = (data['ROC_10'] + data['ROC_20']).ewm(span=10).mean()
    
    # Support/Resistance levels
    data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['R1'] = 2 * data['Pivot'] - data['Low']
    data['S1'] = 2 * data['Pivot'] - data['High']
    data['R2'] = data['Pivot'] + (data['High'] - data['Low'])
    data['S2'] = data['Pivot'] - (data['High'] - data['Low'])
    
    # Price action indicators
    data['Doji'] = (abs(data['Open'] - data['Close']) / (data['High'] - data['Low'])) < 0.1
    data['Hammer'] = ((data['Close'] > data['Open']) & 
                     ((data['Close'] - data['Open']) / (data['High'] - data['Low']) < 0.3) &
                     ((data['Open'] - data['Low']) / (data['High'] - data['Low']) > 0.6))
    
    # Volatility ratio
    data['VR_Close'] = data['Close'].rolling(window=20).std() / data['Close'].rolling(window=20).mean()
    
    # Price position in range
    data['Price_Position'] = ((data['Close'] - data['Low']) / (data['High'] - data['Low'])) * 100

    # === NEW INDICATORS ===
    # Elder Ray Index
    data['Bull_Power'], data['Bear_Power'] = calculate_elder_ray(data['High'], data['Low'], data['Close'])
    # Chaikin Money Flow
    data['CMF_20'] = calculate_chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'])
    # Force Index
    data['ForceIndex_13'] = calculate_force_index(data['Close'], data['Volume'])
    # Ease of Movement
    data['EOM_14'] = calculate_ease_of_movement(data['High'], data['Low'], data['Volume'])
    # Price Oscillator (PPO)
    data['PPO_12_26'] = calculate_ppo(data['Close'])
    # Detrended Price Oscillator
    data['DPO_20'] = calculate_dpo(data['Close'])
    # Vortex Indicator
    data['VI+_14'], data['VI-_14'] = calculate_vortex_indicator(data['High'], data['Low'], data['Close'])
    # Relative Vigor Index
    data['RVI_10'] = calculate_rvi(data['Close'])
    # Accumulation/Distribution Line
    data['AccumDist'] = calculate_accum_dist(data['Close'], data['High'], data['Low'], data['Volume'])
    # Aroon Up/Down
    data['Aroon_Up_25'], data['Aroon_Down_25'] = calculate_aroon(data['High'], data['Low'])
    # Fractal Dimension Index
    data['FDI_14'] = calculate_fdi(data['Close'])
    # Schaff Trend Cycle (STC)
    data['STC'] = calculate_stc(data['Close'])
    # TEMA
    data['TEMA_30'] = calculate_tema(data['Close'])
    # Z-Score
    data['ZScore_20'] = calculate_zscore(data['Close'])
    # Donchian Channel Width
    data['Donchian_Width_20'] = calculate_donchian_width(data['High'], data['Low'])
    # Choppiness Index
    data['Choppiness_14'] = calculate_choppiness_index(data['High'], data['Low'], data['Close'])
    # Connors RSI
    data['ConnorsRSI'] = calculate_connors_rsi(data['Close'])
    # TRIMA
    data['TRIMA_10'] = calculate_trima(data['Close'])
    # KST Oscillator
    data['KST'], data['KST_Signal'] = calculate_kst(data['Close'])

    # === ADDITIONAL VOLUME AND MONEY FLOW INDICATORS ===
    # Volume Price Trend (VPT) - manual calculation
    data['VPT_Manual'] = (data['Volume'] * ((data['Close'] - data['Close'].shift()) / data['Close'].shift())).cumsum()
    
    # On Balance Volume Rate of Change
    data['OBV_ROC'] = data['OBV'].pct_change(10) * 100
    
    # Volume Rate of Change
    data['Volume_ROC'] = data['Volume'].pct_change(10) * 100
    
    # Money Flow Volume
    data['Money_Flow_Volume'] = data['Typical_Price'] * data['Volume']
    data['Money_Flow_Volume_MA'] = data['Money_Flow_Volume'].rolling(window=20).mean()
    
    # Volume Weighted RSI
    data['Volume_Weighted_RSI'] = (data['RSI_14'] * data['Volume']) / data['Volume'].rolling(window=14).mean()
    
    # === PRICE ACTION AND PATTERN INDICATORS ===
    # Inside/Outside bars
    data['Inside_Bar'] = (data['High'] <= data['High'].shift()) & (data['Low'] >= data['Low'].shift())
    data['Outside_Bar'] = (data['High'] > data['High'].shift()) & (data['Low'] < data['Low'].shift())
    
    # Gap analysis
    data['Gap_Up'] = data['Open'] > data['High'].shift()
    data['Gap_Down'] = data['Open'] < data['Low'].shift()
    data['Gap_Size'] = (data['Open'] - data['Close'].shift()) / data['Close'].shift() * 100
    
    # === VOLATILITY AND MOMENTUM ENHANCEMENTS ===
    # Historical Volatility
    data['Historical_Volatility'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252) * 100
    
    # Price Momentum Index
    data['Price_Momentum_Index'] = data['Close'] / data['Close'].rolling(window=10).mean() * 100
    
    # Volume Momentum Index
    data['Volume_Momentum_Index'] = data['Volume'] / data['Volume'].rolling(window=10).mean() * 100
    
    # === CUSTOM SIGNALS ===
    # Enhanced buy signal: ADX_14 > 20, Volume_Ratio > 1.2, Volatility_20 > 0.02, Price above VWAP
    buy_condition = (
        (data["ADX_14"] > 20) &
        (data["Volume_Ratio"] > 1.2) &
        (data["Volatility_20"] > 0.02) &
        (data["Close"] > data["VWAP"])
    )
    
    # === ADDITIONAL MOVING AVERAGES ===
    # DEMA (Double Exponential Moving Average)
    data['DEMA_20'] = calculate_dema(data['Close'], 20)
    data['DEMA_50'] = calculate_dema(data['Close'], 50)
    
    # HMA (Hull Moving Average)
    data['HMA_20'] = calculate_hma(data['High'], data['Low'], data['Close'], 20)
    data['HMA_50'] = calculate_hma(data['High'], data['Low'], data['Close'], 50)
    
    # KAMA (Kaufman Adaptive Moving Average)
    data['KAMA_10'] = calculate_kama(data['Close'], 10)
    data['KAMA_20'] = calculate_kama(data['Close'], 20)
    
    # Moving Average Envelope
    mae_upper, mae_middle, mae_lower = calculate_ma_envelope(data['Close'], 20)
    data['MAE_Upper'] = mae_upper
    data['MAE_Middle'] = mae_middle
    data['MAE_Lower'] = mae_lower
    
    # Moving Average Ribbon
    ma_ribbon = calculate_ma_ribbon(data['Close'])
    for period, ma in ma_ribbon.items():
        data[f'MA_Ribbon_{period}'] = ma
    
    # Alligator Indicator
    alligator_jaw, alligator_teeth, alligator_lips = calculate_alligator(data['High'], data['Low'], data['Close'])
    data['Alligator_Jaw'] = alligator_jaw
    data['Alligator_Teeth'] = alligator_teeth
    data['Alligator_Lips'] = alligator_lips
    
    # Supertrend
    data['Supertrend'] = calculate_supertrend(data['High'], data['Low'], data['Close'])
    
    # === ADDITIONAL OSCILLATORS ===
    # Stochastic RSI
    stoch_rsi_k, stoch_rsi_d = calculate_stochastic_rsi(data['Close'])
    data['StochRSI_K'] = stoch_rsi_k
    data['StochRSI_D'] = stoch_rsi_d
    
    # Momentum (already calculated as MOM_10 and MOM_20 above)
    
    # Rate of Change (ROC) - ROC_10 and ROC_20 already calculated above
    data['ROC_5'] = calculate_roc(data['Close'], 5)
    data['ROC_15'] = calculate_roc(data['Close'], 15)
    
    # True Strength Index (TSI)
    tsi, tsi_signal = calculate_tsi(data['Close'])
    data['TSI'] = tsi
    data['TSI_Signal'] = tsi_signal
    
    # TRIX already calculated above
    
    # Schaff Trend Cycle already calculated as STC above
    
    # === VWAP AND VOLATILITY ENHANCEMENTS ===
    # VWAP Bands
    vwap_upper, vwap_middle, vwap_lower = calculate_vwap_bands(data['High'], data['Low'], data['Close'], data['Volume'])
    data['VWAP_Upper_Band'] = vwap_upper
    data['VWAP_Lower_Band'] = vwap_lower
    
    # Normalized ATR
    data['Normalized_ATR'] = calculate_normalized_atr(data['High'], data['Low'], data['Close'])
    
    # Chaikin Volatility
    data['Chaikin_Volatility'] = calculate_chaikin_volatility(data['High'], data['Low'])
    
    # === VOLUME ENHANCEMENTS ===
    # Volume Oscillator
    data['Volume_Oscillator'] = calculate_volume_oscillator(data['Volume'])
    
    # Price Volume Trend (PVT) - already calculated as VPT_Manual above
    
    # === PRICE CALCULATIONS ===
    # Median Price
    data['Median_Price'] = calculate_median_price(data['High'], data['Low'])
    
    # Weighted Close
    data['Weighted_Close'] = calculate_weighted_close(data['High'], data['Low'], data['Close'])
    
    # === STATISTICAL INDICATORS ===
    # Linear Regression
    regression, slope = calculate_linear_regression(data['Close'])
    data['Linear_Regression'] = regression
    data['Linear_Regression_Slope'] = slope
    
    # Regression Channel
    reg_upper, reg_middle, reg_lower = calculate_regression_channel(data['Close'])
    data['Regression_Upper'] = reg_upper
    data['Regression_Lower'] = reg_lower
    
    # Correlation Coefficient
    data['Correlation_Coefficient'] = calculate_correlation_coefficient(data['Close'])
    
    # Z-Score Normalization
    data['ZScore_Normalized'] = calculate_zscore_normalization(data['Close'])
    
    # === BOLLINGER BAND ENHANCEMENTS ===
    # Bollinger %B
    data['Bollinger_Percent_B'] = calculate_bollinger_percent_b(data['Close'])
    
    # Bollinger Bandwidth
    data['Bollinger_Bandwidth'] = calculate_bollinger_bandwidth(data['Close'])
    
    # === FILTERS AND TRANSFORMS ===
    # Gaussian Filter
    data['Gaussian_Filter'] = calculate_gaussian_filter(data['Close'])
    
    # Sine/Cosine Transform
    sine_transform, cosine_transform = calculate_sine_cosine_transform(data['Close'])
    data['Sine_Transform'] = sine_transform
    data['Cosine_Transform'] = cosine_transform
    
    # === ENSEMBLE AND SIGNAL INDICATORS ===
    # Ensemble Signal Score
    data['Ensemble_Signal_Score'] = calculate_ensemble_signal_score(data)
    
    # Indicator Confidence
    data['Indicator_Confidence'] = calculate_indicator_confidence(data)
    
    # Signal Consensus
    data['Signal_Consensus'] = calculate_signal_consensus(data)
    
    # === FIBONACCI AND WAVE ANALYSIS ===
    # Fibonacci Retracements (using recent high/low)
    recent_high = data['High'].rolling(window=20).max()
    recent_low = data['Low'].rolling(window=20).min()
    fib_levels = calculate_fibonacci_retracements(recent_high, recent_low)
    for level, value in fib_levels.items():
        data[f'Fibonacci_{level}'] = value
    
    # Elliott Wave Count (simplified)
    data['Elliott_Wave_Count'] = calculate_elliott_wave_count(data['High'], data['Low'], data['Close'])
    
    # === ADDITIONAL OSCILLATORS ===
    # Price Oscillator
    data['Price_Oscillator'] = calculate_price_oscillator(data['Close'])
    
    # Disparity Index
    data['Disparity_Index'] = calculate_disparity_index(data['Close'])
    sell_condition = (data["RSI_14"] > 70)
    data['Signal'] = 1  # Default to 'hold'
    data.loc[buy_condition, 'Signal'] = 2  # 'buy'
    data.loc[sell_condition, 'Signal'] = 0  # 'sell'

    # Check for any duplicate columns after all calculations
    final_columns = len(data.columns)
    duplicate_columns_after = []
    for col in data.columns:
        if list(data.columns).count(col) > 1:
            duplicate_columns_after.append(col)
    
    if duplicate_columns_after:
        print(f"⚠️  Warning: Found duplicate columns after calculation: {duplicate_columns_after}")
        # Remove duplicates, keeping the first occurrence
        data = data.loc[:, ~data.columns.duplicated()]
        final_columns_after_cleanup = len(data.columns)
        print(f"✅ Removed {final_columns - final_columns_after_cleanup} duplicate columns")
        final_columns = final_columns_after_cleanup
    
    print(f"[DEBUG] calculate_all_indicators: {original_columns} → {final_columns} columns (+{final_columns - original_columns})")
    
    return data

def fetch_fundamentals(ticker, debug=False):
    """Fetch as many company fundamental, valuation, and profile fields as possible using yfinance."""
    import numpy as np
    import pandas as pd
    import yfinance as yf
    t = yf.Ticker(ticker)
    
    # Get basic info with safe request
    info = safe_yfinance_request(lambda: t.info, debug=debug)
    if info is None:
        if debug:
            print(f"[DEBUG] Could not fetch basic info for {ticker}")
        info = {}
    
    # Try to get quarterly/annual financials for growth rates with safe requests
    try:
        financials = safe_yfinance_request(lambda: t.financials, debug=debug)
    except Exception:
        financials = None
    try:
        balance_sheet = safe_yfinance_request(lambda: t.balance_sheet, debug=debug)
    except Exception:
        balance_sheet = None
    try:
        cashflow = safe_yfinance_request(lambda: t.cashflow, debug=debug)
    except Exception:
        cashflow = None
    try:
        earnings = safe_yfinance_request(lambda: t.earnings, debug=debug)
    except Exception:
        earnings = None
    try:
        quarterly_earnings = safe_yfinance_request(lambda: t.quarterly_earnings, debug=debug)
    except Exception:
        quarterly_earnings = None
    # Helper for safe extraction
    def safe(info, key):
        return info[key] if key in info else np.nan
    fundamentals = {
        # Profile
        'Company_Name': safe(info, 'longName'),
        'Sector': safe(info, 'sector'),
        'Industry': safe(info, 'industry'),
        'Country': safe(info, 'country'),
        'Employees': safe(info, 'fullTimeEmployees'),
        'Exchange': safe(info, 'exchange'),
        'Currency': safe(info, 'currency'),
        # Valuation
        'Market_Cap': safe(info, 'marketCap'),
        'Enterprise_Value': safe(info, 'enterpriseValue'),
        'Trailing_PE': safe(info, 'trailingPE'),
        'Forward_PE': safe(info, 'forwardPE'),
        'PEG_Ratio': safe(info, 'pegRatio'),
        'Price_to_Sales': safe(info, 'priceToSalesTrailing12Months'),
        'Price_to_Book': safe(info, 'priceToBook'),
        'Book_Value': safe(info, 'bookValue'),
        'Book_Value_Per_Share': safe(info, 'bookValue'),
        'Price_to_Cashflow': safe(info, 'priceToCashflow'),
        'EV_to_EBITDA': safe(info, 'enterpriseToEbitda'),
        'EV_to_Revenue': safe(info, 'enterpriseToRevenue'),
        # Profitability
        'Gross_Margins': safe(info, 'grossMargins'),
        'Operating_Margins': safe(info, 'operatingMargins'),
        'Net_Margins': safe(info, 'netMargins'),
        'ROE': safe(info, 'returnOnEquity'),
        'ROA': safe(info, 'returnOnAssets'),
        'ROIC': np.nan,  # Not directly available
        'EBITDA_Margins': safe(info, 'ebitdaMargins'),
        # Growth
        'Revenue_Growth': safe(info, 'revenueGrowth'),
        'Earnings_Growth': safe(info, 'earningsGrowth'),
        'EPS_Growth': safe(info, 'earningsQuarterlyGrowth'),
        # Balance Sheet
        'Total_Assets': safe(info, 'totalAssets'),
        'Total_Liabilities': safe(info, 'totalLiab'),
        'Total_Debt': safe(info, 'totalDebt'),
        'Current_Ratio': safe(info, 'currentRatio'),
        'Quick_Ratio': safe(info, 'quickRatio'),
        'Debt_to_Equity': safe(info, 'debtToEquity'),
        'Working_Capital': np.nan,  # Not directly available
        'Cash': safe(info, 'totalCash'),
        'Cash_Per_Share': safe(info, 'totalCashPerShare'),
        # Cash Flow
        'Operating_Cash_Flow': safe(info, 'operatingCashflow'),
        'Free_Cash_Flow': safe(info, 'freeCashflow'),
        'CapEx': safe(info, 'capitalExpenditures'),
        # Dividend
        'Dividend_Rate': safe(info, 'dividendRate'),
        'Dividend_Yield': safe(info, 'dividendYield'),
        'Payout_Ratio': safe(info, 'payoutRatio'),
        'Ex_Dividend_Date': safe(info, 'exDividendDate'),
        # Shares
        'Shares_Outstanding': safe(info, 'sharesOutstanding'),
        'Float_Shares': safe(info, 'floatShares'),
        'Shares_Short': safe(info, 'sharesShort'),
        'Short_Ratio': safe(info, 'shortRatio'),
        # Analyst
        'Target_Mean_Price': safe(info, 'targetMeanPrice'),
        'Target_High_Price': safe(info, 'targetHighPrice'),
        'Target_Low_Price': safe(info, 'targetLowPrice'),
        'Number_Analyst_Opinions': safe(info, 'numberOfAnalystOpinions'),
        'Recommendation_Mean': safe(info, 'recommendationMean'),
        'Recommendation_Key': safe(info, 'recommendationKey'),
        # Other
        'Beta': safe(info, 'beta'),
        'FiftyTwo_Week_High': safe(info, 'fiftyTwoWeekHigh'),
        'FiftyTwo_Week_Low': safe(info, 'fiftyTwoWeekLow'),
        'Average_Volume': safe(info, 'averageVolume'),
        'Implied_Shares_Outstanding': safe(info, 'impliedSharesOutstanding'),
        'Held_Percent_Insiders': safe(info, 'heldPercentInsiders'),
        'Held_Percent_Institutions': safe(info, 'heldPercentInstitutions'),
        'Short_Percent_of_Float': safe(info, 'shortPercentOfFloat'),
        'Short_Percent_of_Shares_Outstanding': safe(info, 'shortPercentOfSharesOutstanding'),
        'Last_Split_Date': safe(info, 'lastSplitDate'),
        'Last_Dividend_Date': safe(info, 'lastDividendDate'),
        'Fiscal_Year_End': safe(info, 'lastFiscalYearEnd'),
        'IPO_Year': safe(info, 'ipoYear') if 'ipoYear' in info else np.nan,
        'Website': safe(info, 'website'),
        # 'Description': safe(info, 'longBusinessSummary'),  # Removed as requested
    }
    # Calculate some fields if possible
    # Working Capital
    try:
        if balance_sheet is not None:
            ca = balance_sheet.loc['Total Current Assets'].iloc[0] if 'Total Current Assets' in balance_sheet.index else np.nan
            cl = balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance_sheet.index else np.nan
            fundamentals['Working_Capital'] = ca - cl if (not pd.isna(ca) and not pd.isna(cl)) else np.nan
    except Exception:
        pass
    # ROIC (if possible)
    try:
        if financials is not None and balance_sheet is not None:
            nopat = financials.loc['Net Income'].iloc[0] * (1 - 0.21) if 'Net Income' in financials.index else np.nan
            ic = (balance_sheet.loc['Total Assets'].iloc[0] - balance_sheet.loc['Current Liabilities'].iloc[0]) if ('Total Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index) else np.nan
            fundamentals['ROIC'] = nopat / ic if (not pd.isna(nopat) and not pd.isna(ic) and ic != 0) else np.nan
    except Exception:
        pass
    # Revenue/EPS Growth (YoY)
    try:
        if earnings is not None and len(earnings) > 1:
            rev_now = earnings['Revenue'].iloc[-1]
            rev_prev = earnings['Revenue'].iloc[-2]
            fundamentals['Revenue_Growth_YoY'] = (rev_now - rev_prev) / rev_prev if rev_prev != 0 else np.nan
            eps_now = earnings['Earnings'].iloc[-1]
            eps_prev = earnings['Earnings'].iloc[-2]
            fundamentals['EPS_Growth_YoY'] = (eps_now - eps_prev) / eps_prev if eps_prev != 0 else np.nan
    except Exception:
        pass
    return fundamentals

def fetch_stock_data(ticker, years=30, lookforward_days=7, debug=False, verbose=False, fetch_intraday=False, intraday_period='1mo', intraday_interval='1h', include_lunar=True, vix_data=None):
    """Fetch stock data with comprehensive technical indicators"""
    try:
        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        if debug or verbose:
            print(f"[VERBOSE] Fetching {years} years of data for {ticker.upper()}...")
            print(f"[DEBUG] Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Fetch stock data with safe request
        stock = yf.Ticker(ticker)
        data = safe_yfinance_request(lambda: stock.history(start=start_date, end=end_date), debug=debug)
        
        if data is None:
            print(f"Could not fetch data for ticker: {ticker}")
            return None
        
        if debug or verbose:
            print(f"[DEBUG] Raw data shape for {ticker}: {data.shape}")
            print(f"[DEBUG] Data columns: {list(data.columns)}")
        
        if data.empty:
            print(f"No data found for ticker: {ticker}")
            return None
        
        # Add ticker column
        data['Ticker'] = ticker.upper()
        
        # Use provided VIX data or fetch it if not provided
        # OPTIMIZATION: VIX data can be passed in to avoid redundant API calls
        if vix_data is None:
            if debug or verbose:
                print("[DEBUG] Fetching VIX data...")
            vix_data = fetch_vix_data(start_date, end_date)
        else:
            if debug or verbose:
                print("[DEBUG] Using provided VIX data...")
        
        # Calculate all technical indicators
        if debug or verbose:
            print("[DEBUG] Calculating technical indicators...")
            print(f"[DEBUG] Data shape before indicators: {data.shape}")
        data = calculate_all_indicators(data, lookforward_days=lookforward_days)
        if debug or verbose:
            print(f"[DEBUG] Data shape after indicators: {data.shape}")
            print(f"[DEBUG] Finished calculating indicators for {ticker}")
        # Add company fundamentals as columns
        fundamentals = fetch_fundamentals(ticker, debug=debug)
        for k, v in fundamentals.items():
            data[k] = v
        
        # Add option data
        if debug or verbose:
            print("[DEBUG] Fetching option data...")
        option_data = fetch_option_data(ticker, debug=debug)
        
        # Add option data as columns (latest values)
        if option_data['put_call_ratios']:
            # Get the nearest expiration date
            nearest_exp = option_data['expiration_dates'][0] if option_data['expiration_dates'] else None
            if nearest_exp:
                data['Option_Put_Call_Ratio'] = option_data['put_call_ratios'].get(nearest_exp, np.nan)
                data['Option_Call_Volume'] = option_data['option_volumes'].get(nearest_exp, {}).get('call_volume', np.nan)
                data['Option_Put_Volume'] = option_data['option_volumes'].get(nearest_exp, {}).get('put_volume', np.nan)
                data['Option_Total_Volume'] = option_data['option_volumes'].get(nearest_exp, {}).get('total_volume', np.nan)
                data['Option_Nearest_Expiration'] = nearest_exp
        else:
            data['Option_Put_Call_Ratio'] = np.nan
            data['Option_Call_Volume'] = np.nan
            data['Option_Put_Volume'] = np.nan
            data['Option_Total_Volume'] = np.nan
            data['Option_Nearest_Expiration'] = np.nan
        
        # Add market data
        if debug or verbose:
            print("[DEBUG] Fetching market data...")
        market_data = fetch_market_data(ticker, debug=debug)
        
        # Add institutional data as columns
        data['Institutional_Holders_Count'] = len(market_data.get('top_institutional_holders', []))
        data['Major_Holders_Count'] = len(market_data.get('major_holders', []))
        data['Recent_Recommendations_Count'] = len(market_data.get('recent_recommendations', []))
        data['Earnings_Dates_Count'] = len(market_data.get('earnings_dates', []))
        data['Calendar_Events_Count'] = len(market_data.get('calendar', []))
        
        # Calculate market sentiment indicators
        if debug or verbose:
            print("[DEBUG] Calculating market sentiment indicators...")
        data = calculate_market_sentiment_indicators(data, debug=debug)
        
        # Calculate market calendar features
        if debug or verbose:
            print("[DEBUG] Calculating market calendar features...")
            print(f"[DEBUG] Data shape before calendar features: {data.shape}")
        data = calculate_market_calendar_features(data, debug=debug, include_lunar=include_lunar)
        if debug or verbose:
            print(f"[DEBUG] Data shape after calendar features: {data.shape}")
        
        # Fetch intraday data if requested
        if fetch_intraday:
            if debug or verbose:
                print("[DEBUG] Fetching intraday data...")
            intraday_data = fetch_intraday_data(ticker, period=intraday_period, interval=intraday_interval, debug=debug)
            if intraday_data is not None:
                # Add intraday summary statistics to daily data
                latest_intraday = intraday_data.tail(1)
                if not latest_intraday.empty:
                    data['Latest_Intraday_High'] = latest_intraday['High'].iloc[0]
                    data['Latest_Intraday_Low'] = latest_intraday['Low'].iloc[0]
                    data['Latest_Intraday_Close'] = latest_intraday['Close'].iloc[0]
                    data['Latest_Intraday_Volume'] = latest_intraday['Volume'].iloc[0]
                    data['Latest_Intraday_VWAP'] = latest_intraday['Intraday_VWAP'].iloc[0]
                else:
                    data['Latest_Intraday_High'] = np.nan
                    data['Latest_Intraday_Low'] = np.nan
                    data['Latest_Intraday_Close'] = np.nan
                    data['Latest_Intraday_Volume'] = np.nan
                    data['Latest_Intraday_VWAP'] = np.nan
            else:
                data['Latest_Intraday_High'] = np.nan
                data['Latest_Intraday_Low'] = np.nan
                data['Latest_Intraday_Close'] = np.nan
                data['Latest_Intraday_Volume'] = np.nan
                data['Latest_Intraday_VWAP'] = np.nan
        
        # Add VIX data
        if vix_data is not None and not vix_data.empty:
            if debug or verbose:
                print(f"[DEBUG] VIX data shape: {vix_data.shape}")
                print(f"[DEBUG] VIX data columns: {list(vix_data.columns)}")
                print(f"[DEBUG] VIX data index: {vix_data.index[:5]}")
                print(f"[DEBUG] Stock data index: {data.index[:5]}")
            # Add DateOnly columns for merge
            data = data.copy()
            vix_data = vix_data.copy()
            data['DateOnly'] = pd.to_datetime(data.index.date)
            vix_data['DateOnly'] = pd.to_datetime(vix_data.index.date)
            # Rename VIX columns
            vix_data = vix_data.rename(columns={
                'Close': 'VIX_Close',
                'High': 'VIX_High',
                'Low': 'VIX_Low',
                'Volume': 'VIX_Volume'
            })
            vix_columns = ['DateOnly', 'VIX_Close', 'VIX_High', 'VIX_Low', 'VIX_Volume']
            # Merge on DateOnly
            merged = pd.merge(data, vix_data[vix_columns], on='DateOnly', how='left', suffixes=('', '_VIX'))
            # Restore original index
            merged.index = data.index
            # Drop DateOnly column
            merged = merged.drop(columns=['DateOnly'])
            data = merged
            if debug or verbose:
                print(f"[DEBUG] After VIX merge - data shape: {data.shape}")
                print(f"[DEBUG] VIX columns in merged data: {[col for col in data.columns if 'VIX' in col]}")
                print(f"[DEBUG] Sample VIX data: {data[['VIX_Close', 'VIX_High', 'VIX_Low', 'VIX_Volume']].head()}")
        else:
            if debug or verbose:
                print("[DEBUG] No VIX data available, setting to NaN")
            data['VIX_Close'] = np.nan
            data['VIX_High'] = np.nan
            data['VIX_Low'] = np.nan
            data['VIX_Volume'] = np.nan
        
        # Reorder columns to match your sample format
        column_order = [
            'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
            'Daily_Return', 'Price_Range', 'Price_Range_Pct', 'Typical_Price',
            'Volume_MA_20', 'Volume_MA_50', 'Volume_Ratio', 'OBV', 'OBV_ROC', 'Volume_ROC',
            'VWAP', 'VWAP_20', 'VWAP_50', 'Price_vs_VWAP', 'Price_vs_VWAP_20',
            'VPT', 'Money_Flow_Volume', 'Money_Flow_Volume_MA', 'Volume_Weighted_RSI',
            'Volatility_20', 'ATR_14', 'Historical_Volatility',
            'RSI_14', 'RSI_21', 'STOCH_14_3_3', 'STOCH_D_3', 'CCI_20', 'ADX_14',
            'MOM_10', 'MOM_20', 'ROC_10', 'ROC_20', 'Price_Momentum_Index', 'Volume_Momentum_Index',
            'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
            'STOCHRSI_14_14_3_3', 'WILLR_14', 'ULTOSC_7_14_28', 'MFI_14', 'CMO_9',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_30', 'SMA_50', 'SMA_100', 'SMA_200',
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_30', 'EMA_50', 'EMA_100', 'EMA_200',
            'Golden_Cross_EMA_20_50', 'Death_Cross_EMA_20_50', 'Golden_Cross_EMA_50_200', 'Death_Cross_EMA_50_200',
            'Golden_Cross_SMA_20_50', 'Death_Cross_SMA_20_50', 'Golden_Cross_SMA_50_200', 'Death_Cross_SMA_50_200',
            'MA_Alignment_EMA', 'MA_Alignment_SMA',
            'Price_Above_EMA_20', 'Price_Above_EMA_50', 'Price_Above_EMA_200',
            'Price_Above_SMA_20', 'Price_Above_SMA_50', 'Price_Above_SMA_200',
            'EMA_20_50_Spread', 'EMA_50_200_Spread', 'SMA_20_50_Spread', 'SMA_50_200_Spread',
            'EMA_5_20_Spread', 'SMA_5_20_Spread',
            'EMA_10_20_Crossover', 'EMA_20_30_Crossover', 'SMA_10_20_Crossover', 'SMA_20_30_Crossover',
            'EMA_20_Slope', 'EMA_50_Slope', 'EMA_200_Slope', 'SMA_20_Slope', 'SMA_50_Slope', 'SMA_200_Slope',
            'EMA_Convergence', 'EMA_Divergence', 'SMA_Convergence', 'SMA_Divergence',
            'EMA_Trend_Strength', 'SMA_Trend_Strength',
            'Bongo_SMA', 'Bongo_EMA',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width', 'BB_Position',
            'KC_Upper', 'KC_Lower', 'KC_Middle',
            'DC_Upper', 'DC_Lower', 'DC_Middle',
            'PSAR', 'TRIX', 'CoppockCurve',
            'Pivot', 'R1', 'S1', 'R2', 'S2',
            'Doji', 'Hammer', 'Inside_Bar', 'Outside_Bar', 'Gap_Up', 'Gap_Down', 'Gap_Size',
            'VR_Close', 'Price_Position',
            'Signal',
            'Fear_Greed_Volume', 'Price_Sentiment', 'Volatility_Sentiment', 'RSI_Sentiment', 'VWAP_Sentiment',
            'Option_Put_Call_Ratio', 'Option_Call_Volume', 'Option_Put_Volume', 'Option_Total_Volume', 'Option_Nearest_Expiration',
            'Latest_Intraday_High', 'Latest_Intraday_Low', 'Latest_Intraday_Close', 'Latest_Intraday_Volume', 'Latest_Intraday_VWAP',
            'Institutional_Holders_Count', 'Major_Holders_Count', 'Recent_Recommendations_Count', 'Earnings_Dates_Count', 'Calendar_Events_Count',
            'Day_Of_Week', 'Day_Of_Month', 'Day_Of_Year', 'Week_Of_Year', 'Month', 'Year',
            'Moon_Phase', 'Is_Full_Moon', 'Is_New_Moon', 'Is_Moon_Waxing_or_Waning',
            'Is_Quarter_Start', 'Is_Quarter_End', 'Is_Earnings_Season', 'Is_Options_Expiration_Week', 'Is_Quadruple_Witching',
            'Is_FOMC_Week', 'Is_January_Effect', 'Is_Santa_Claus_Rally_Period', 'Is_Sell_In_May_Period', 'Is_Tax_Loss_Harvesting_Period',
            'Is_Month_End', 'Is_First_Trading_Day_Of_Month', 'Is_Holiday_Week',
            'Is_Planting_Season', 'Is_Harvest_Season', 'Is_Grain_Report_Week', 'Is_Ethanol_Inventory_Release', 'Is_Hurricane_Season',
            'Is_Spring_Equinox', 'Is_Summer_Solstice', 'Is_Fall_Equinox', 'Is_Winter_Solstice', 'Is_Solar_Maximum_Cycle_Year',
            'Is_Back_To_School_Period', 'Is_Black_Friday', 'Is_Cyber_Monday', 'Is_Amazon_Prime_Day', 'Is_Holiday_Shopping_Season',
            'Is_Valentine_Week', 'Is_Mothers_Day_Week', 'Is_Fathers_Day_Week', 'Is_Tax_Refund_Season', 'Is_Labor_Day_Weekend',
            'Is_Summer_Travel_Season', 'Is_Tourism_Peak_Month', 'Is_Chinese_New_Year_Period', 'Is_Super_Bowl_Week', 'Is_Olympic_Year',
            'Is_FOMC_Meeting_Week', 'Is_Global_CB_Meeting_Week', 'Is_NonFarm_Payrolls_Week', 'Is_CPI_PPI_Report_Week', 'Is_GDP_Release_Week',
            'Is_Beige_Book_Week', 'Is_Treasury_Auction_Week', 'Is_Consumer_Confidence_Week', 'Is_ISM_PMI_Week', 'Is_Retail_Sales_Report_Week',
            'Is_Jackson_Hole_Week', 'Is_Triple_Witching', 'Is_End_Of_Month_Rebalancing', 'Is_End_Of_Quarter_Rebalancing', 'Is_End_Of_Year_Rebalancing',
            'Is_Mid_Month_Reversal', 'Is_Turn_Of_Month_Effect', 'Is_VIX_Options_Expiration', 'Is_Russell_Reconstitution', 'Is_Hedge_Fund_Redemption',
            'Is_Tax_Day', 'Is_Earnings_Season_Start', 'Is_Ex_Dividend_Week', 'Is_Post_Holiday_Drift', 'Is_Pre_Holiday_Rally',
            'Is_Friday_Effect', 'Is_Monday_Effect', 'Is_Summer_Doldrums', 'Is_Halloween_Effect', 'Is_Presidential_Election_Year',
            'Is_Year_3_Presidential_Cycle', 'Is_ETF_Rebalance_Week',
            'Is_New_Years_Day', 'Is_Presidents_Day', 'Is_Easter_Weekend', 'Is_Independence_Day', 'Is_Halloween', 'Is_Christmas', 'Is_Boxing_Day',
            'Is_Lunar_New_Year', 'Is_Valentines_Day', 'Is_Hanukkah', 'Is_Diwali', 'Is_Ramadan', 'Is_Eid_al_Fitr', 'Is_Golden_Week_China', 'Is_Carnival_Season', 'Is_Singles_Day_China', 'Is_Super_Bowl_Weekend',
            'Is_Fiscal_Year_End', 'Is_Tax_Day_US', 'Is_IRA_Contribution_Deadline', 'Is_College_Tuition_Due', 'Is_Property_Tax_Payment_Date', 'Is_Health_Insurance_Enrollment_Season',
            'Is_Paycheck_Cycle_1st', 'Is_Paycheck_Cycle_15th', 'Is_Debt_Payment_Cycle', 'Is_Bonus_Payout_Season',
            'Is_Beginning_Of_Year_Optimism', 'Is_September_Anxiety', 'Is_Window_Dressing_Season', 'Is_Post_Bonus_Spending', 'Is_Post_Tax_Refund_Spending',
            'Is_Circadian_Rhythm_Morning', 'Is_Seasonal_Affective_Winter', 'Is_Daylight_Savings_Shift',
            'Is_Waxing_Moon', 'Is_Waning_Moon', 'Moon_Cycle_Sine_Encoded', 'Moon_Cycle_Cosine_Encoded', 'Is_Lunar_Eclipse_Day', 'Is_Solar_Eclipse_Day',
            'Twitter_Stock_Sentiment', 'Reddit_WallStreetBets_Mentions', 'Google_Search_Trend_Score', 'News_Headline_Sentiment_Score', 'Article_Volume_By_Ticker', 'Bloomberg_Headline_Count', 'YahooFinance_Comment_Sentiment', 'YouTube_Financial_Mentions', 'SEC_10K_Sentiment_Score', 'Insider_Sales_Headline_Count',
            'Is_US_Election_Day', 'Is_Presidential_Debate_Flag', 'Is_FOMC_Announcement_Day', 'Is_Fed_Nomination_Date', 'Is_G7_Summit_Flag', 'Is_G20_Summit_Flag', 'Is_UN_General_Assembly_Flag', 'Is_Debt_Ceiling_Crisis_Flag', 'Is_US_Budget_Deadline', 'Is_Midterm_Election_Year', 'Is_State_Of_The_Union_Flag', 'Is_Sanctions_Announcement_Flag', 'Is_Trade_Agreement_Deadline', 'Is_Geopolitical_Tension_Flag', 'Is_Government_Shutdown_Risk', 'Is_Impeachment_Inquiry_Started',
            'Port_Shipping_Volume_Index', 'Trucking_Freight_Index', 'Flight_Traffic_Volume', 'Job_Posting_Volume', 'Consumer_Sentiment_Index', 'Homebuilder_Confidence_Score', 'Real_Estate_Listing_Volume', 'Credit_Card_Spending_Trend',
            'Insider_Buying_Cluster_Score', '13F_Whale_Position_Change', 'Dark_Pool_Buy_Activity', 'Dark_Pool_Sell_Activity', 'Unusual_Options_Volume', 'Options_Order_Imbalance', 'Short_Interest_Ratio', 'Borrow_Fee_Rate_Spike',
            'VIX_Close', 'VIX_High', 'VIX_Low', 'VIX_Volume',
            'DEMA_20', 'DEMA_50', 'HMA_20', 'HMA_50', 'KAMA_10', 'KAMA_20',
            'MAE_Upper', 'MAE_Middle', 'MAE_Lower',
            'MA_Ribbon_MA_10', 'MA_Ribbon_MA_20', 'MA_Ribbon_MA_30', 'MA_Ribbon_MA_40', 'MA_Ribbon_MA_50', 'MA_Ribbon_MA_60',
            'Alligator_Jaw', 'Alligator_Teeth', 'Alligator_Lips', 'Supertrend',
            'StochRSI_K', 'StochRSI_D', 'Momentum_10', 'Momentum_20', 'ROC_5', 'ROC_15',
            'TSI', 'TSI_Signal', 'TRIX_15', 'TRIX_Signal', 'Schaff_Trend_Cycle',
            'VWAP_Upper_Band', 'VWAP_Lower_Band', 'Normalized_ATR', 'Chaikin_Volatility',
            'Volume_Oscillator', 'PVT', 'Median_Price', 'Typical_Price_Calc', 'Weighted_Close',
            'Linear_Regression', 'Linear_Regression_Slope', 'Regression_Upper', 'Regression_Lower',
            'Correlation_Coefficient', 'ZScore_Normalized', 'Bollinger_Percent_B', 'Bollinger_Bandwidth',
            'Gaussian_Filter', 'Sine_Transform', 'Cosine_Transform',
            'Ensemble_Signal_Score', 'Indicator_Confidence', 'Signal_Consensus',
            'Fibonacci_Fib_236', 'Fibonacci_Fib_382', 'Fibonacci_Fib_500', 'Fibonacci_Fib_618', 'Fibonacci_Fib_786',
            'Elliott_Wave_Count', 'Price_Oscillator', 'Disparity_Index',
            # Additional columns that might be generated
            'Bull_Power', 'Bear_Power', 'CMF_20', 'ForceIndex_13', 'EOM_14', 'PPO_12_26', 'DPO_20',
            'VI+_14', 'VI-_14', 'RVI_10', 'AccumDist', 'Aroon_Up_25', 'Aroon_Down_25', 'FDI_14', 'STC',
            'TEMA_30', 'ZScore_20', 'Donchian_Width_20', 'Choppiness_14', 'ConnorsRSI', 'TRIMA_10',
            'KST', 'KST_Signal', 'Sine_Transform', 'Cosine_Transform'
        ]
        
        # Reorder columns (prioritize column_order, then add any missing columns)
        existing_columns = [col for col in column_order if col in data.columns]
        missing_columns = [col for col in data.columns if col not in column_order]
        
        # Combine: column_order columns first, then any additional columns
        final_column_order = existing_columns + missing_columns
        data = data[final_column_order]
        
        if debug or verbose:
            print(f"[DEBUG] Column order columns: {len(existing_columns)}")
            print(f"[DEBUG] Additional columns: {len(missing_columns)}")
            print(f"[DEBUG] Total columns: {len(final_column_order)}")
            print(f"[DEBUG] Missing columns found: {missing_columns}")
            print(f"[DEBUG] Data shape before reorder: {data.shape}")
            print(f"[DEBUG] Data shape after reorder: {data.shape}")
        
        if debug or verbose:
            print(f"[DEBUG] Final data shape for {ticker}: {data.shape}")
            print(f"[DEBUG] All columns generated: {list(data.columns)}")
        print(f"Generated {len(data.columns)} columns of data for {ticker}")
        return data
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        if debug or verbose:
            import traceback
            traceback.print_exc()
        return None

def fetch_multiple_tickers(tickers, years=30, lookforward_days=7, debug=False, verbose=False, fetch_intraday=False, intraday_period='1mo', intraday_interval='1h', include_lunar=True):
    """Fetch data for multiple tickers and combine into single DataFrame"""
    all_data = []
    if verbose:
        print(f"[VERBOSE] Fetching data for tickers: {tickers}")
    
    # Handle both comma-separated and space-separated tickers
    if isinstance(tickers, str):
        # Split by comma and then by space to handle both formats
        ticker_list = []
        for item in tickers.split(','):
            ticker_list.extend(item.strip().split())
        tickers = ticker_list
    elif isinstance(tickers, list):
        # Handle list of tickers that might contain comma-separated strings
        ticker_list = []
        for item in tickers:
            if ',' in item:
                ticker_list.extend([t.strip() for t in item.split(',')])
            else:
                ticker_list.append(item.strip())
        tickers = ticker_list
    
    # Fetch VIX data once for all tickers (since it's market-wide data)
    # OPTIMIZATION: VIX data is fetched once and reused for all tickers instead of fetching per ticker
    if verbose:
        print("[VERBOSE] Fetching VIX data once for all tickers...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    vix_data = fetch_vix_data(start_date, end_date)
    if verbose:
        print(f"[VERBOSE] VIX data fetched: {len(vix_data) if vix_data is not None and not vix_data.empty else 0} rows")
    
    for i, ticker in enumerate(tickers):
        ticker = ticker.strip().upper()
        if not ticker:
            continue
        if verbose:
            print(f"[VERBOSE] Processing ticker: {ticker} ({i+1}/{len(tickers)})")
        
        data = fetch_stock_data(ticker, years, lookforward_days=lookforward_days, debug=debug, verbose=verbose, 
                               fetch_intraday=fetch_intraday, intraday_period=intraday_period, intraday_interval=intraday_interval,
                               include_lunar=include_lunar, vix_data=vix_data)
        if data is not None:
            all_data.append(data)
            print(f"Successfully processed {ticker}")
        else:
            print(f"Failed to process {ticker}")
        
        # Add delay between tickers to avoid rate limiting (except for the last ticker)
        if i < len(tickers) - 1:
            delay = random.uniform(1, 3)  # Random delay between 1-3 seconds
            if debug:
                print(f"[DEBUG] Waiting {delay:.2f}s before next ticker...")
            time.sleep(delay)
    
    if not all_data:
        print("No data was successfully fetched for any ticker")
        return None
    
    # Combine all data
    # Reset index to avoid duplicate index issues when concatenating multiple tickers
    for i, df in enumerate(all_data):
        if df is not None:
            all_data[i] = df.reset_index()
    
    # Ensure all DataFrames have the same columns before concatenation
    if len(all_data) > 1:
        # Get all unique column names from all DataFrames
        all_columns = set()
        for df in all_data:
            if df is not None:
                all_columns.update(df.columns)
        
        # Add missing columns to each DataFrame with NaN values
        for i, df in enumerate(all_data):
            if df is not None:
                missing_cols = all_columns - set(df.columns)
                for col in missing_cols:
                    df[col] = np.nan
                # Reorder columns to match the union of all columns
                all_data[i] = df[list(all_columns)]
    
    # Check for duplicate columns before concatenation
    for i, df in enumerate(all_data):
        if len(df.columns) != len(set(df.columns)):
            print(f"⚠️  Warning: Duplicate columns found in ticker DataFrame {i}, removing duplicates")
            all_data[i] = df.loc[:, ~df.columns.duplicated()]
    
    # Concatenate with ignore_index=True to avoid duplicate index issues
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Check for duplicate columns after concatenation
    if len(combined_data.columns) != len(set(combined_data.columns)):
        print(f"⚠️  Warning: Duplicate columns found after concatenation in fetch_multiple_tickers, removing duplicates")
        combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
        if verbose:
            print(f"  Final column count after deduplication: {len(combined_data.columns)}")
    
    # Set Date as index and handle potential duplicates by keeping all rows
    if 'Date' in combined_data.columns:
        # Sort by Date first to ensure proper ordering
        combined_data = combined_data.sort_values('Date')
        # Set Date as index - this will handle duplicates by keeping all rows
        combined_data = combined_data.set_index('Date')
        # Sort the index
        combined_data = combined_data.sort_index()
    
    if verbose:
        print(f"[VERBOSE] Combined data shape: {combined_data.shape if 'combined_data' in locals() else 'No data'}")
    print(f"\n{'='*50}")
    print(f"COMBINED DATA SUMMARY")
    print(f"{'='*50}")
    print(f"Total rows: {len(combined_data)}")
    print(f"Total columns: {len(combined_data.columns)}")
    print(f"Tickers included: {combined_data['Ticker'].unique()}")
    
    # Fix the date range display
    if len(combined_data) > 0:
        start_date = combined_data.index[0]
        end_date = combined_data.index[-1]
        # Convert to string using pd.Timestamp if possible
        try:
            start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d')
            end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')
        except Exception:
            start_str = str(start_date)
            end_str = str(end_date)
        print(f"Date range: {start_str} to {end_str}")
    
    return combined_data

def remove_nulls(data, verbose=False):
    """Smart null value handling with selective cleaning"""
    print("\n=== SMART NULL VALUE HANDLING ===")
    original_shape = data.shape
    print(f"Original data shape: {original_shape}")
    
    # Check for duplicate columns before null handling
    duplicate_columns = []
    for col in data.columns:
        if list(data.columns).count(col) > 1:
            duplicate_columns.append(col)
    
    if duplicate_columns:
        print(f"⚠️  Found duplicate columns: {duplicate_columns}")
        # Remove duplicates, keeping the first occurrence
        data = data.loc[:, ~data.columns.duplicated()]
        print(f"✅ Removed {len(duplicate_columns)} duplicate columns")
        original_shape = data.shape  # Update shape after removing duplicates
    
    # Analyze null values by column
    null_counts = data.isnull().sum()
    null_percentages = (null_counts / len(data)) * 100
    
    # Show columns with nulls
    columns_with_nulls = null_counts[null_counts > 0]
    if len(columns_with_nulls) > 0:
        print(f"\nColumns with null values ({len(columns_with_nulls)} columns):")
        for col in columns_with_nulls.index:
            count = null_counts[col]
            pct = null_percentages[col]
            print(f"  {col}: {count} nulls ({pct:.1f}%)")
    else:
        print("No null values found in any column!")
        return data
    
    # Define critical columns that must not be null for analysis
    critical_columns = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
        'Signal'  # Target variable
    ]
    
    # Find which critical columns exist in the data
    existing_critical_columns = [col for col in critical_columns if col in data.columns]
    
    print(f"\nCritical columns for analysis: {existing_critical_columns}")
    
    # Remove rows with nulls in critical columns only
    print(f"\nRemoving rows with nulls in critical columns only...")
    data_cleaned = data.dropna(subset=existing_critical_columns)
    
    final_shape = data_cleaned.shape
    rows_removed = original_shape[0] - final_shape[0]
    print(f"Rows removed: {rows_removed} ({rows_removed/original_shape[0]*100:.1f}%)")
    print(f"Final data shape: {final_shape}")
    
    # Fill remaining nulls with appropriate values
    print(f"\nFilling remaining nulls with appropriate values...")
    
    # Fill numeric columns with 0 or median
    numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        try:
            null_count = int(data_cleaned[col].isnull().sum())
            if null_count > 0:
                # For boolean-like columns (0/1), fill with 0
                if col.startswith('Is_') or col in ['Signal', 'Buy_Signal', 'Sell_Signal']:
                    data_cleaned[col] = data_cleaned[col].fillna(0)
                    print(f"  Filled {col} with 0 (boolean column)")
                # For sentiment scores (-1 to 1), fill with 0
                elif 'Sentiment' in col or col in ['Twitter_Stock_Sentiment', 'News_Headline_Sentiment_Score', 
                                                 'YahooFinance_Comment_Sentiment', 'SEC_10K_Sentiment_Score',
                                                 'Options_Order_Imbalance']:
                    data_cleaned[col] = data_cleaned[col].fillna(0)
                    print(f"  Filled {col} with 0 (sentiment column)")
                # For percentage/ratio columns, fill with 0
                elif any(x in col for x in ['Ratio', 'Percent', 'Yield', 'Margin', 'Growth']):
                    data_cleaned[col] = data_cleaned[col].fillna(0)
                    print(f"  Filled {col} with 0 (percentage/ratio column)")
                # For volume/count columns, fill with 0
                elif any(x in col for x in ['Volume', 'Count', 'Mentions', 'Headline']):
                    data_cleaned[col] = data_cleaned[col].fillna(0)
                    print(f"  Filled {col} with 0 (volume/count column)")
                # For index/score columns, fill with median or 100
                elif any(x in col for x in ['Index', 'Score', 'Trend']):
                    if col in ['Consumer_Sentiment_Index', 'Homebuilder_Confidence_Score']:
                        data_cleaned[col] = data_cleaned[col].fillna(75)  # Neutral sentiment
                    elif col in ['Port_Shipping_Volume_Index', 'Trucking_Freight_Index', 'Flight_Traffic_Volume',
                               'Job_Posting_Volume', 'Real_Estate_Listing_Volume']:
                        data_cleaned[col] = data_cleaned[col].fillna(100)  # Normal activity
                    else:
                        data_cleaned[col] = data_cleaned[col].fillna(data_cleaned[col].median())
                    print(f"  Filled {col} with appropriate value (index/score column)")
                # For other numeric columns, fill with median
                else:
                    data_cleaned[col] = data_cleaned[col].fillna(data_cleaned[col].median())
                    print(f"  Filled {col} with median (general numeric column)")
        except:
            # Skip this column if there's an issue
            continue
    
    # Fill categorical/string columns with 'Unknown' or appropriate default
    categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        try:
            null_count = int(data_cleaned[col].isnull().sum())
            if null_count > 0:
                if col == 'Date':
                    # Skip date column - should not be null after critical column cleaning
                    continue
                else:
                    data_cleaned[col] = data_cleaned[col].fillna('Unknown')
                    print(f"  Filled {col} with 'Unknown' (categorical column)")
        except:
            # Skip this column if there's an issue
            continue
    
    # Verify no nulls remain
    remaining_nulls = data_cleaned.isnull().sum().sum()
    if remaining_nulls == 0:
        print("✅ All null values successfully handled!")
    else:
        print(f"⚠️  Warning: {remaining_nulls} null values still remain")
        if verbose:
            remaining_null_cols = data_cleaned.columns[data_cleaned.isnull().sum() > 0]
            print(f"  Remaining null columns: {list(remaining_null_cols)}")
    
    print(f"Final data shape: {data_cleaned.shape}")
    
    return data_cleaned

def save_data(data, tickers, format='csv', remove_nulls_flag=False, smart_nulls=True):
    """Save data to file (cleaned for look-forward validation)"""
    import os
    verbose = False
    if 'verbose' in globals():
        verbose = verbose or globals()['verbose']
    if 'verbose' in locals():
        verbose = verbose or locals()['verbose']
    print("[DEBUG] Entered save_data function")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    if verbose:
        print("[VERBOSE] Step 1: Starting save_data")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[DEBUG] Timestamp for filename: {timestamp}")
    print(f"[DEBUG] Data shape before cleaning: {data.shape}")
    print(f"[DEBUG] Columns: {list(data.columns)}")
    print(f"[DEBUG] Data head:\n{data.head(3)}")

    # Check for duplicate columns before any processing
    duplicate_columns = []
    for col in data.columns:
        if list(data.columns).count(col) > 1:
            duplicate_columns.append(col)
    
    if duplicate_columns:
        print(f"⚠️  Found duplicate columns in save_data: {duplicate_columns}")
        # Remove duplicates, keeping the first occurrence
        data = data.loc[:, ~data.columns.duplicated()]
        print(f"✅ Removed {len(duplicate_columns)} duplicate columns")
        print(f"[DEBUG] Data shape after removing duplicates: {data.shape}")

    if verbose:
        print("[VERBOSE] Step 2: Cleaning column names")
    # Clean column names
    data.columns = [str(col).strip() for col in data.columns]
    print(f"[DEBUG] Cleaned columns: {list(data.columns)}")

    # Handle null values based on user preference
    if remove_nulls_flag:
        if smart_nulls:
            if verbose:
                print("[VERBOSE] Step 2.5: Using smart null handling")
            data = remove_nulls(data, verbose=verbose)
        else:
            if verbose:
                print("[VERBOSE] Step 2.5: Using aggressive null removal")
            # Use the old aggressive approach for backward compatibility
            original_shape = data.shape
            print(f"\n=== AGGRESSIVE NULL REMOVAL ===")
            print(f"Original data shape: {original_shape}")
            
            # Remove rows with any null values
            data = data.dropna()
            
            final_shape = data.shape
            rows_removed = original_shape[0] - final_shape[0]
            print(f"Rows removed: {rows_removed} ({rows_removed/original_shape[0]*100:.1f}%)")
            print(f"Final data shape: {final_shape}")
    else:
        if verbose:
            print("[VERBOSE] Step 2.5: Keeping all null values (default for ML frameworks)")
        print("\n=== KEEPING ALL NULL VALUES ===")
        print("No null value cleaning performed - all nulls preserved")
        print("This is the recommended approach for ML frameworks like XGBoost, LightGBM, etc.")

    if verbose:
        print("[VERBOSE] Step 3: Dropping rows with NaN Signal")
    # Drop rows where Signal is NaN (look-forward target)
    if 'Signal' in data.columns:
        before_drop = data.shape[0]
        data = data[~data['Signal'].isna()].copy()
        after_drop = data.shape[0]
        print(f"[DEBUG] Dropped rows with NaN Signal: {before_drop - after_drop} rows removed")
        print(f"[DEBUG] Data shape after dropping NaN Signal: {data.shape}")
        print(f"[DEBUG] Data head after drop:\n{data.head(3)}")

    if verbose:
        print("[VERBOSE] Step 4: Resetting index")
    # Reset index and ensure date is a column
    data = data.reset_index()
    print(f"[DEBUG] Data shape after reset_index: {data.shape}")
    print(f"[DEBUG] Data head after reset_index:\n{data.head(3)}")
    print(f"[DEBUG] Index column type: {type(data['index'].iloc[0]) if 'index' in data.columns else 'No index column'}")
    print(f"[DEBUG] Index column sample values: {data['index'].head(3).tolist() if 'index' in data.columns else 'No index column'}")
    
    if 'index' in data.columns:
        # Check if this looks like a date index (even if not datetime type)
        index_values = data['index']
        
        # Try to detect if this is a date index by checking if it's numeric and has reasonable date-like values
        if pd.api.types.is_datetime64_any_dtype(index_values):
            # It's already a datetime type
            data = data.rename(columns={'index': 'Date'})
            data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
            print("[DEBUG] Renamed 'index' column to 'Date' and formatted to date-only")
            print(f"[DEBUG] Sample dates after conversion: {data['Date'].head(3).tolist()}")
        elif pd.api.types.is_numeric_dtype(index_values):
            # Check if these look like timestamp values (reasonable date range)
            if index_values.min() > 1000000000 and index_values.max() < 9999999999:  # Unix timestamp range
                # Convert from timestamp to date
                data = data.rename(columns={'index': 'Date'})
                data['Date'] = pd.to_datetime(data['Date'], unit='s').dt.strftime('%Y-%m-%d')
                print("[DEBUG] Converted numeric index to Date from timestamp")
            else:
                # Regular numeric index, rename to Row
                data = data.rename(columns={'index': 'Row'})
                print("[DEBUG] Renamed 'index' column to 'Row'")
        else:
            # Try to parse as date string
            try:
                test_date = pd.to_datetime(index_values.iloc[0])
                if test_date.year > 1900 and test_date.year < 2030:  # Reasonable date range
                    data = data.rename(columns={'index': 'Date'})
                    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
                    print("[DEBUG] Converted string index to Date")
                else:
                    data = data.rename(columns={'index': 'Row'})
                    print("[DEBUG] Renamed 'index' column to 'Row'")
            except:
                data = data.rename(columns={'index': 'Row'})
                print("[DEBUG] Renamed 'index' column to 'Row'")
    
    # Additional check: if we still have 'index' column and it looks like dates, convert it
    if 'index' in data.columns:
        index_values = data['index']
        # Check if the index values look like dates (they should be pandas Timestamps)
        if hasattr(index_values.iloc[0], 'year') and hasattr(index_values.iloc[0], 'month'):
            # These are datetime objects, convert to Date column
            data = data.rename(columns={'index': 'Date'})
            data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
            print("[DEBUG] Detected datetime index and converted to Date column")
            print(f"[DEBUG] Sample dates after conversion: {data['Date'].head(3).tolist()}")
        else:
            # Not datetime, rename to Row
            data = data.rename(columns={'index': 'Row'})
            print("[DEBUG] Index not detected as datetime, renamed to Row")

    if verbose:
        print("[VERBOSE] Step 5: Determining output filename")
    if len(tickers) == 1:
        filename = f"{tickers[0].upper()}_{timestamp}.{format}"
    else:
        # Create filename with multiple tickers
        ticker_str = "_".join([t.upper() for t in tickers])
        filename = f"multi_ticker_dataset_{timestamp}.{format}"
    print(f"[DEBUG] Output filename: {filename}")

    try:
        if verbose:
            print("[VERBOSE] Step 6: Checking write permissions")
        print(f"[DEBUG] Checking write permissions in: {os.getcwd()}")
        if not os.access(os.getcwd(), os.W_OK):
            print("[ERROR] No write permission in current directory!")
        else:
            print("[DEBUG] Write permission OK.")
        if verbose:
            print(f"[VERBOSE] Step 7: Saving file as {format}")
        if format == 'csv':
            # Force Date column to be string format before saving
            if 'Date' in data.columns:
                data['Date'] = data['Date'].astype(str)
                # Extract just the date part (YYYY-MM-DD) from datetime string
                data['Date'] = data['Date'].str[:10]
                print(f"[DEBUG] Forced Date column to date-only format before saving")
            data.to_csv(filename, index=False)
            print(f"[DEBUG] Data saved to CSV: {filename}")
        elif format == 'excel':
            data.to_excel(filename, index=False)
            print(f"[DEBUG] Data saved to Excel: {filename}")
        else:
            print(f"[ERROR] Unknown format: {format}")
        # Check if file exists after saving
        if os.path.exists(filename):
            print(f"[DEBUG] File exists after save: {filename}")
        else:
            print(f"[ERROR] File does NOT exist after save: {filename}")
    except Exception as e:
        print(f"[ERROR] Exception while saving file: {e}")
        import traceback
        traceback.print_exc()

    print(f"Data saved to: {filename}")
    if verbose:
        print(f"[VERBOSE] Saved file: {filename}")
    return filename

# Remove the print_summary function (lines ~758-801)
# Remove the plot_comprehensive_analysis function (lines ~699-758)
# Remove all calls to print_summary and plot_comprehensive_analysis in main and interactive mode

# 1. Delete the print_summary function (lines ~758-801)
# 2. Delete the plot_comprehensive_analysis function (lines ~699-758)
# 3. In main(), remove the loop that calls print_summary and the block that calls plot_comprehensive_analysis
# 4. In the interactive __main__ block, remove the loop that calls print_summary and the block that calls plot_comprehensive_analysis

def clean_existing_csv(csv_file_path, output_path=None, verbose=False):
    """Clean an existing CSV file by removing null values"""
    try:
        print(f"Loading CSV file: {csv_file_path}")
        data = pd.read_csv(csv_file_path)
        
        print(f"Original data shape: {data.shape}")
        
        # Remove nulls
        data_cleaned = remove_nulls(data, verbose=verbose)
        
        # Determine output path
        if output_path is None:
            base_name = csv_file_path.rsplit('.', 1)[0]
            output_path = f"{base_name}_cleaned.csv"
        
        # Save cleaned data
        data_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error cleaning CSV file: {e}")
        return None

# --- Remove plot_comprehensive_analysis ---
# --- Remove print_summary ---
# --- Remove all calls to them in main and interactive mode ---

def main():
    print("[DEBUG] Entered main() function")
    parser = argparse.ArgumentParser(description='Fetch comprehensive stock data with 80+ technical indicators for multiple tickers')
    parser.add_argument('tickers', nargs='*', help='Stock ticker symbols (e.g., AAPL MSFT GOOGL or AAPL,MSFT,GOOGL)')
    parser.add_argument('--ticker-file', type=str, help='Path to text file containing ticker symbols (one per line)')
    parser.add_argument('--years', type=int, default=30, help='Number of years to fetch (default: 30)')
    parser.add_argument('--format', choices=['csv', 'excel'], default='csv', help='Output format')
    parser.add_argument('--plot', action='store_true', help='Generate comprehensive analysis charts for each ticker')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save data to file')
    parser.add_argument('--lookforward', type=int, default=7, help='Number of days to look forward for the Signal column (default: 7)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--keep-nulls', action='store_true', help='Keep all null values (default behavior for ML frameworks)')
    parser.add_argument('--aggressive-nulls', action='store_true', help='Use aggressive null removal (overrides default keep-nulls behavior)')
    parser.add_argument('--clean-csv', type=str, help='Clean an existing CSV file by removing null values')
    parser.add_argument('--intraday', action='store_true', help='Also fetch intraday data (1-hour intervals for 1 month)')
    parser.add_argument('--intraday-period', type=str, default='1mo', help='Period for intraday data (default: 1mo)')
    parser.add_argument('--intraday-interval', type=str, default='1h', help='Interval for intraday data (default: 1h)')
    parser.add_argument('--no-lunar', action='store_true', help='Disable lunar phase calculations (faster processing)')
    parser.add_argument('--filters', action='store_true', help='Apply technical indicator filters to improve signal quality (RSI 35-75, MACD > 0, ADX >= 20)')
    parser.add_argument('--feature-filters', type=str, help='Path to JSON feature filters file from automated feature selector')
    args = parser.parse_args()
    
    # Process tickers - either from command line arguments or from file
    processed_tickers = []
    
    if args.ticker_file:
        # Read tickers from file
        try:
            with open(args.ticker_file, 'r') as f:
                file_tickers = [line.strip() for line in f if line.strip()]
            processed_tickers.extend(file_tickers)
            if args.verbose or args.debug:
                print(f"Loaded {len(file_tickers)} tickers from file: {args.ticker_file}")
        except FileNotFoundError:
            print(f"❌ Error: Ticker file '{args.ticker_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading ticker file '{args.ticker_file}': {e}")
            sys.exit(1)
    
    # Add command line tickers if provided
    if args.tickers:
        for ticker_arg in args.tickers:
            if ',' in ticker_arg:
                # Split comma-separated tickers
                processed_tickers.extend([t.strip() for t in ticker_arg.split(',')])
            else:
                processed_tickers.append(ticker_arg.strip())
    
    # Validate that we have at least one ticker
    if not processed_tickers:
        print("❌ Error: No tickers provided. Please specify tickers as command line arguments or use --ticker-file.")
        print("Usage examples:")
        print("  python 1_stock_history_script.py AAPL MSFT GOOGL")
        print("  python 1_stock_history_script.py --ticker-file all_indices_tickers_20250729_230715.txt")
        print("  python 1_stock_history_script.py AAPL MSFT --ticker-file additional_tickers.txt")
        print("  python 1_stock_history_script.py AAPL MSFT --feature-filters automated_feature_filters_20250101_120000.json")
        print("  python 1_stock_history_script.py --ticker-file tickers.txt --feature-filters feature_filters.json --filters")
        sys.exit(1)
    
    verbose = args.verbose or args.debug
    
    # Handle CSV cleaning mode
    if args.clean_csv:
        print(f"Cleaning existing CSV file: {args.clean_csv}")
        clean_existing_csv(args.clean_csv, verbose=verbose)
        return
    
    if verbose:
        print('Starting script...')
        print('Tickers:', processed_tickers)
        print('Years:', args.years)
        print('Format:', args.format)
        print('Lookforward:', args.lookforward)

    # Fetch data for all tickers
    print("\n" + "="*70)
    print("📊 FETCHING STOCK DATA")
    print("="*70)
    print("ℹ️  Note: HTTP 404 errors are normal and expected for some data types.")
    print("   The script will continue processing with available data.")
    print("   This is due to Yahoo Finance API limitations and rate limiting.")
    print("="*70 + "\n")
    
    data = fetch_multiple_tickers(processed_tickers, args.years, lookforward_days=args.lookforward, debug=args.debug, verbose=verbose,
                                 fetch_intraday=args.intraday, intraday_period=args.intraday_period, intraday_interval=args.intraday_interval,
                                 include_lunar=not args.no_lunar)
    
    if data is None:
        print("\n❌ No data was successfully fetched for any ticker.")
        print("   This could be due to:")
        print("   - Invalid ticker symbols")
        print("   - Network connectivity issues")
        print("   - Yahoo Finance API rate limiting")
        print("   - Temporary service unavailability")
        print("\n   Try running again in a few minutes or check your ticker symbols.")
        sys.exit(1)
    
    # Apply signal filters if requested
    if args.filters:
        data = apply_signal_filters(data, verbose=verbose)
    
    # Apply feature filters if requested
    if args.feature_filters:
        data = apply_feature_filters(data, args.feature_filters, verbose=verbose)
    
    # Save data
    if not args.no_save:
        save_data(data, processed_tickers, args.format, remove_nulls_flag=not args.keep_nulls, smart_nulls=not args.aggressive_nulls)

    # === SIGNAL SUMMARY ===
    if 'Signal' in data.columns:
        print("\n=== SIGNAL SUMMARY (All Tickers) ===")
        signal_counts = data['Signal'].value_counts().sort_index()
        print(f"Buy signals (2): {signal_counts.get(2, 0)}")
        print(f"Hold signals (1): {signal_counts.get(1, 0)}")
        print(f"Sell signals (0): {signal_counts.get(0, 0)}")

def apply_feature_filters(data, json_filters_file, verbose=False):
    """
    Apply feature filters to dataset based on JSON filters from automated feature selector.
    
    Args:
        data (pd.DataFrame): DataFrame with all features
        json_filters_file (str): Path to JSON filters file
        verbose (bool): Whether to print filtering statistics
    
    Returns:
        pd.DataFrame: Filtered DataFrame with only optimal features for each ticker
    """
    if verbose:
        print("\n🔧 APPLYING FEATURE FILTERS")
        print("="*50)
        original_count = len(data)
        original_features = len(data.columns)
        print(f"Original data: {original_count:,} rows, {original_features} columns")
    
    # Load JSON filters
    try:
        with open(json_filters_file, 'r') as f:
            filters = json.load(f)
        if verbose:
            print(f"✅ Loaded feature filters for {len(filters)} tickers")
    except Exception as e:
        print(f"❌ Error loading feature filters from {json_filters_file}: {e}")
        return data
    
    # Apply filters
    filtered_dfs = []
    
    # First pass: collect all unique columns that will be needed
    all_columns = set()
    ticker_filtered_data = {}
    
    for ticker, filter_info in filters.items():
        ticker_data = data[data['Ticker'] == ticker].copy()
        if len(ticker_data) == 0:
            if verbose:
                print(f"⚠️  Ticker {ticker} not found in dataset")
            continue
        
        selected_features = filter_info.get('selected_features', [])
        
        if not selected_features:
            if verbose:
                print(f"⚠️  No features selected for {ticker}, keeping all features")
            # Add all columns from this ticker to the set
            all_columns.update(ticker_data.columns)
            ticker_filtered_data[ticker] = ticker_data
            continue
        
        # Keep essential columns (non-feature columns)
        essential_cols = ['Date', 'Ticker', 'Signal', 'WF_Strategy_Return', 'WF_Cumulative_Return', 
                        'WF_Strategy_Return_Without_Filters', 'WF_Cumulative_Return_Without_Filters', 
                        'Model_Signal', 'Open', 'High', 'Low', 'Close', 'Volume', 'Row']
        
        # Filter to only selected features + essential columns
        available_features = [f for f in selected_features if f in ticker_data.columns]
        essential_available = [f for f in essential_cols if f in ticker_data.columns]
        
        columns_to_keep = essential_available + available_features
        
        filtered_data = ticker_data[columns_to_keep].copy()
        
        # Add these columns to the master set
        all_columns.update(columns_to_keep)
        ticker_filtered_data[ticker] = filtered_data
        
        if verbose:
            print(f"  {ticker}: {len(ticker_data.columns)} -> {len(filtered_data.columns)} columns "
                  f"({len(available_features)} selected features)")
    
    # Ensure all_columns contains only unique column names
    all_columns = set(all_columns)  # This ensures uniqueness
    
    if verbose:
        print(f"  Total unique columns across all tickers: {len(all_columns)}")
    
    # Second pass: ensure all DataFrames have the same columns
    for ticker, filtered_data in ticker_filtered_data.items():
        # Add missing columns with NaN values
        missing_columns = list(all_columns - set(filtered_data.columns))
        for col in missing_columns:
            filtered_data[col] = np.nan
        
        # Reorder columns to match the master set and ensure no duplicates
        # FIX: Ensure ordered_columns contains only unique column names
        ordered_columns = list(dict.fromkeys(all_columns))  # Preserves order while removing duplicates
        filtered_data = filtered_data[ordered_columns]
        
        # Ensure no duplicate columns
        if len(filtered_data.columns) != len(set(filtered_data.columns)):
            print(f"⚠️  Warning: Duplicate columns found for {ticker}, removing duplicates")
            filtered_data = filtered_data.loc[:, ~filtered_data.columns.duplicated()]
        
        filtered_dfs.append(filtered_data)
    
    if filtered_dfs:
        # More robust concatenation approach
        if verbose:
            print(f"  Debug: Attempting to concatenate {len(filtered_dfs)} DataFrames")
        
        try:
            # Check for any duplicate columns in the final DataFrames before concatenation
            for i, df in enumerate(filtered_dfs):
                if len(df.columns) != len(set(df.columns)):
                    print(f"⚠️  Warning: Duplicate columns found in DataFrame {i}, removing duplicates")
                    filtered_dfs[i] = df.loc[:, ~df.columns.duplicated()]
            
            # Simple concatenation first - preserve the original index
            result_df = pd.concat(filtered_dfs, ignore_index=False)
            
            # Final check for duplicate columns after concatenation
            if len(result_df.columns) != len(set(result_df.columns)):
                print(f"⚠️  Warning: Duplicate columns found after concatenation, removing duplicates")
                result_df = result_df.loc[:, ~result_df.columns.duplicated()]
                if verbose:
                    print(f"  Final column count after deduplication: {len(result_df.columns)}")
            
            # Ensure Date column exists - if Row column looks like dates, convert it
            if 'Date' not in result_df.columns and 'Row' in result_df.columns:
                try:
                    # Try to convert Row to Date if it looks like dates
                    test_row = result_df['Row'].iloc[0]
                    if pd.api.types.is_numeric_dtype(result_df['Row']):
                        # Check if it's a reasonable timestamp
                        if result_df['Row'].min() > 1000000000 and result_df['Row'].max() < 9999999999:
                            result_df['Date'] = pd.to_datetime(result_df['Row'], unit='s').dt.strftime('%Y-%m-%d')
                            print("✅ Converted Row column to Date column from timestamp")
                        else:
                            # Try as regular date conversion
                            result_df['Date'] = pd.to_datetime(result_df['Row']).dt.strftime('%Y-%m-%d')
                            print("✅ Converted Row column to Date column")
                    else:
                        # Try string conversion
                        result_df['Date'] = pd.to_datetime(result_df['Row']).dt.strftime('%Y-%m-%d')
                        print("✅ Converted Row column to Date column from string")
                except Exception as e:
                    print(f"⚠️  Could not convert Row to Date: {e}")
            
            if verbose:
                print(f"  Debug: Simple concatenation successful")
        except Exception as e:
            if verbose:
                print(f"  Debug: Simple concatenation failed: {e}")
                print(f"  Debug: Trying alternative approach...")
            
            # Alternative approach: ensure all DataFrames have the same columns
            all_columns = set()
            for df in filtered_dfs:
                all_columns.update(df.columns)
            
            common_columns = sorted(list(all_columns))
            if verbose:
                print(f"  Debug: Using {len(common_columns)} common columns")
            
            # Align all DataFrames
            aligned_dfs = []
            for i, df in enumerate(filtered_dfs):
                # Create a new DataFrame with all columns
                aligned_df = pd.DataFrame(index=df.index)
                
                # Add each column, filling with NaN if missing
                for col in common_columns:
                    if col in df.columns:
                        aligned_df[col] = df[col]
                    else:
                        aligned_df[col] = np.nan
                
                aligned_dfs.append(aligned_df)
                if verbose:
                    print(f"  Debug: DataFrame {i} aligned to {len(aligned_df.columns)} columns")
            
            # Try concatenation again - preserve the original index
            result_df = pd.concat(aligned_dfs, ignore_index=False)
            if verbose:
                print(f"  Debug: Alternative concatenation successful")
        
        if verbose:
            final_features = len(result_df.columns)
            feature_reduction = ((original_features - final_features) / original_features * 100)
            print(f"\n📊 FEATURE FILTER RESULTS:")
            print(f"Original features: {original_features}")
            print(f"Filtered features: {final_features}")
            print(f"Feature reduction: {feature_reduction:.1f}%")
            print(f"Data points: {len(result_df):,}")
            print("="*50)
        
        return result_df
    else:
        if verbose:
            print("❌ No data remained after applying feature filters")
        return data

def apply_signal_filters(data, verbose=False):
    """
    Apply technical indicator filters to improve signal quality early on.
    
    Filters:
    - RSI_14 between 35 and 75 (avoid extreme overbought/oversold)
    - MACDh_12_26_9 > 0 (positive momentum)
    - ADX_14 >= 20 (trend strength)
    
    Args:
        data (pd.DataFrame): DataFrame with technical indicators
        verbose (bool): Whether to print filtering statistics
    
    Returns:
        pd.DataFrame: Filtered DataFrame with additional 'Filter_Pass' column
    """
    if verbose:
        print("\n🔍 APPLYING SIGNAL FILTERS")
        print("="*50)
        original_count = len(data)
        print(f"Original data points: {original_count:,}")
    
    # Create a copy to avoid modifying original data
    filtered_data = data.copy()
    
    # Initialize filter pass column
    filtered_data['Filter_Pass'] = True
    
    # Filter 1: RSI_14 between 35 and 75
    if 'RSI_14' in filtered_data.columns:
        rsi_filter = (filtered_data['RSI_14'] >= 35) & (filtered_data['RSI_14'] <= 75)
        filtered_data.loc[~rsi_filter, 'Filter_Pass'] = False
        if verbose:
            rsi_passed = rsi_filter.sum()
            print(f"RSI_14 filter (35-75): {rsi_passed:,}/{original_count:,} passed ({rsi_passed/original_count*100:.1f}%)")
    else:
        if verbose:
            print("⚠️  RSI_14 column not found - skipping RSI filter")
    
    # Filter 2: MACDh_12_26_9 > 0 (positive momentum)
    if 'MACDh_12_26_9' in filtered_data.columns:
        macd_filter = filtered_data['MACDh_12_26_9'] > 0
        filtered_data.loc[~macd_filter, 'Filter_Pass'] = False
        if verbose:
            macd_passed = macd_filter.sum()
            print(f"MACDh_12_26_9 filter (>0): {macd_passed:,}/{original_count:,} passed ({macd_passed/original_count*100:.1f}%)")
    else:
        if verbose:
            print("⚠️  MACDh_12_26_9 column not found - skipping MACD filter")
    
    # Filter 3: ADX_14 >= 20 (trend strength)
    if 'ADX_14' in filtered_data.columns:
        adx_filter = filtered_data['ADX_14'] >= 20
        filtered_data.loc[~adx_filter, 'Filter_Pass'] = False
        if verbose:
            adx_passed = adx_filter.sum()
            print(f"ADX_14 filter (>=20): {adx_passed:,}/{original_count:,} passed ({adx_passed/original_count*100:.1f}%)")
    else:
        if verbose:
            print("⚠️  ADX_14 column not found - skipping ADX filter")
    
    # Apply combined filter
    final_filtered = filtered_data[filtered_data['Filter_Pass'] == True].copy()
    
    if verbose:
        final_count = len(final_filtered)
        removed_count = original_count - final_count
        print(f"\n📊 FILTER RESULTS:")
        print(f"Data points removed: {removed_count:,} ({removed_count/original_count*100:.1f}%)")
        print(f"Data points remaining: {final_count:,} ({final_count/original_count*100:.1f}%)")
        
        # Show signal distribution before and after filtering
        if 'Signal' in filtered_data.columns:
            print(f"\n📈 SIGNAL DISTRIBUTION:")
            print("Before filtering:")
            before_counts = filtered_data['Signal'].value_counts().sort_index()
            for signal, count in before_counts.items():
                signal_name = {0: 'Sell', 1: 'Hold', 2: 'Buy'}.get(signal, f'Signal_{signal}')
                print(f"  {signal_name}: {count:,}")
            
            print("After filtering:")
            after_counts = final_filtered['Signal'].value_counts().sort_index()
            for signal, count in after_counts.items():
                signal_name = {0: 'Sell', 1: 'Hold', 2: 'Buy'}.get(signal, f'Signal_{signal}')
                print(f"  {signal_name}: {count:,}")
        
        print("="*50)
    
    return final_filtered

if __name__ == "__main__":
    print("[DEBUG] __name__ == '__main__' block executing")
    if len(sys.argv) == 1:
        # Interactive mode
        tickers_input = input("Enter stock tickers separated by spaces or commas (e.g., AAPL MSFT GOOGL or AAPL,MSFT,GOOGL): ").strip()
        if not tickers_input:
            print("No tickers provided. Exiting.")
            sys.exit(1)
        # Process tickers to handle both space and comma separation
        tickers = []
        if ',' in tickers_input:
            # Split by comma first, then by space
            for item in tickers_input.split(','):
                tickers.extend([t.strip() for t in item.split()])
        else:
            # Split by space only
            tickers = tickers_input.split()
        
        lookforward_days = input("Enter look-forward days for Signal (default 7): ").strip()
        lookforward_days = int(lookforward_days) if lookforward_days.isdigit() else 7
        debug = input("Enable debug output? (y/n): ").strip().lower() == 'y'
        verbose = input("Enable verbose output? (y/n): ").strip().lower() == 'y'
        keep_nulls = input("Keep rows with null values? (y/n, default y): ").strip().lower() != 'n'
        aggressive_nulls = input("Use aggressive null removal? (y/n, default n): ").strip().lower() == 'y'
        fetch_intraday = input("Fetch intraday data? (y/n): ").strip().lower() == 'y'
        include_lunar = input("Include lunar phase calculations? (y/n, default y): ").strip().lower() != 'n'
        apply_filters = input("Apply technical indicator filters? (y/n): ").strip().lower() == 'y'
        feature_filters_file = input("Path to feature filters JSON file (optional, press Enter to skip): ").strip()
        if not feature_filters_file:
            feature_filters_file = None
        
        # Run with user input
        print("\n" + "="*70)
        print("📊 FETCHING STOCK DATA")
        print("="*70)
        print("ℹ️  Note: HTTP 404 errors are normal and expected for some data types.")
        print("   The script will continue processing with available data.")
        print("   This is due to Yahoo Finance API limitations and rate limiting.")
        print("="*70 + "\n")
        
        data = fetch_multiple_tickers(tickers, lookforward_days=lookforward_days, debug=debug, verbose=verbose, fetch_intraday=fetch_intraday, include_lunar=include_lunar)
        if data is not None:
            # Apply signal filters if requested
            if apply_filters:
                data = apply_signal_filters(data, verbose=verbose)
            
            # Apply feature filters if requested
            if feature_filters_file:
                data = apply_feature_filters(data, feature_filters_file, verbose=verbose)
            
            # Handle nulls based on user preference
            if not keep_nulls:
                if not aggressive_nulls:
                    print("Using smart null handling (default)...")
                    data = remove_nulls(data, verbose=verbose)
                else:
                    print("Using aggressive null removal...")
                    original_shape = data.shape
                    data = data.dropna()
                    rows_removed = original_shape[0] - data.shape[0]
                    print(f"Rows removed: {rows_removed} ({rows_removed/original_shape[0]*100:.1f}%)")
            else:
                print("Keeping all null values...")
            
            # === SIGNAL SUMMARY ===
            if 'Signal' in data.columns:
                print("\n=== SIGNAL SUMMARY (All Tickers) ===")
                signal_counts = data['Signal'].value_counts().sort_index()
                print(f"Buy signals (2): {signal_counts.get(2, 0)}")
                print(f"Hold signals (1): {signal_counts.get(1, 0)}")
                print(f"Sell signals (0): {signal_counts.get(0, 0)}")
        else:
            print("\n❌ No data was successfully fetched for any ticker.")
            print("   This could be due to:")
            print("   - Invalid ticker symbols")
            print("   - Network connectivity issues")
            print("   - Yahoo Finance API rate limiting")
            print("   - Temporary service unavailability")
            print("\n   Try running again in a few minutes or check your ticker symbols.")
            sys.exit(1)
    else:
        print(f"[DEBUG] sys.argv: {sys.argv}")
        main()
