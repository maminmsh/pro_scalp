import pandas as pd
import numpy as np
from ta.trend import ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator, WilliamsRIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import config


# ─────────────────────────────────────────────
# 1. HELPER FUNCTIONS
# Purpose: Utility functions to mimic Pine Script behavior
# ─────────────────────────────────────────────
def f_nr(src, is_confirmed):
    """Forces series to update only on confirmed (closed) bars."""
    return src if is_confirmed else src.shift(1).fillna(method='ffill')


def f_tr(high, low, close):
    """Calculates True Range."""
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    return np.maximum(tr1, np.maximum(tr2, tr3))


def f_tanh(x):
    """Custom tanh function for Fisher transform."""
    e2x = np.exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)


def hma(src, length):
    """Hull Moving Average."""
    wma1 = src.rolling(window=length).mean()
    wma2 = src.rolling(window=int(length / 2)).mean()
    diff = 2 * wma2 - wma1
    return diff.rolling(window=int(np.sqrt(length))).mean()


# ─────────────────────────────────────────────
# 2. DATA PREPARATION
# Purpose: Prepare confirmed OHLCV data
# ─────────────────────────────────────────────
def prepare_data(df, is_confirmed=True):
    """Prepares confirmed OHLCV data."""
    cOpen = f_nr(df['open'], is_confirmed)
    cHigh = f_nr(df['high'], is_confirmed)
    cLow = f_nr(df['low'], is_confirmed)
    cClose = f_nr(df['close'], is_confirmed)
    cVolume = f_nr(df['volume'], is_confirmed)
    return cOpen, cHigh, cLow, cClose, cVolume


# ─────────────────────────────────────────────
# 3. INDICATOR CALCULATIONS
# Purpose: Calculate trend, volatility, momentum, and volume indicators
# ─────────────────────────────────────────────

# 3.1 ADX & DI Calculation
def calculate_adx_di(high, low, close, adx_len, di_len):
    """Calculates ADX and DI values."""
    adx_indicator = ADXIndicator(high=high, low=low, close=close, window=di_len)
    diplus = adx_indicator.adx_pos()
    diminus = adx_indicator.adx_neg()
    adx_val = adx_indicator.adx()
    trend_dir = np.where(diplus > diminus, 1.0, np.where(diminus > diplus, -1.0, 0.0))
    norm_trend = trend_dir * np.minimum(np.maximum((adx_val - 20) / 20, 0.0), 1.0)
    return f_nr(adx_val, True), f_nr(norm_trend, True)


# 3.2 Volatility: ATR
def calculate_atr(high, low, close, atr_len):
    """Calculates normalized ATR."""
    atr_indicator = AverageTrueRange(high=high, low=low, close=close, window=atr_len)
    atr_val = atr_indicator.average_true_range()
    norm_atr = atr_val / close
    norm_volatility_raw = norm_atr * 2 - 1
    return f_nr(norm_volatility_raw, True)


# 3.3 Momentum Calculations
def calculate_stoch_rsi(close, length, k, d):
    """Calculates Stochastic RSI."""
    rsi = RSIIndicator(close=close, window=length).rsi()
    lowest_val = rsi.rolling(window=length).min()
    highest_val = rsi.rolling(window=length).max()
    stoch_val = (rsi - lowest_val) / np.maximum(highest_val - lowest_val, 0.0001)
    k_val = stoch_val.rolling(window=k).mean()
    return (k_val - 0.5) * 2


def calculate_fisher_rsi(close, length):
    """Calculates Fisher RSI."""
    rsi = RSIIndicator(close=close, window=length).rsi()
    norm_rsi = (rsi / 100.0) * 2 - 1
    clipped = np.clip(norm_rsi, -0.999, 0.999)
    fish = 0.5 * np.log((1 + clipped) / (1 - clipped))
    return f_tanh(fish)


def calculate_wpr(high, low, close, length):
    """Calculates Williams %R."""
    wpr = WilliamsRIndicator(high=high, low=low, close=close, lbp=length).williams_r()
    return (wpr + 50) / 50 - 1


def calculate_momentum(close, high, low, mom_choice):
    """Calculates momentum based on selected mode."""
    if mom_choice == "Stochastic RSI":
        return calculate_stoch_rsi(close, config.stochRsiLen, config.stochKLen, config.stochDLen)
    elif mom_choice == "Fisher RSI":
        return calculate_fisher_rsi(close, config.fisherRsiLen)
    else:  # Williams %R
        return calculate_wpr(high, low, close, config.wprLen)


# 3.4 TDI (Optional Reference)
def calculate_tdi(close, rsi_len, signal_len):
    """Calculates TDI RSI and Signal."""
    tdi_rsi = RSIIndicator(close=close, window=rsi_len).rsi()
    tdi_signal = EMAIndicator(close=tdi_rsi, window=signal_len).ema_indicator()
    return f_nr(tdi_rsi, True), f_nr(tdi_signal, True)


# 3.5 Volume (OBV Calculation)
def calculate_obv(close, volume):
    """Calculates normalized OBV."""
    obv = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    obv_min = obv.rolling(window=config.volLookback).min()
    obv_max = obv.rolling(window=config.volLookback).max()
    norm_volume_raw = ((obv - obv_min) / np.maximum(obv_max - obv_min, 0.0001)) * 2 - 1
    return f_nr(norm_volume_raw, True)


# 3.6 Dynamic Weighting
def calculate_dynamic_weights(adx_val, norm_atr):
    """Calculates dynamic weights for trend and volatility."""
    if config.useDynamicW:
        base_vol = 0.005
        ratio_vol = norm_atr / base_vol
        vol_factor = np.minimum(ratio_vol, config.maxVolFactor)
        base_adx = 25.0
        ratio_adx = adx_val / base_adx
        trend_factor = np.minimum(ratio_adx, config.maxTrendFactor)
    else:
        vol_factor = 1.0
        trend_factor = 1.0
    eff_w_trend = config.wTrend * trend_factor
    eff_w_volatility = config.wVolatility * vol_factor
    eff_w_momentum = config.wMomentum
    eff_w_volume = config.wVolume
    total_weight = eff_w_trend + eff_w_momentum + eff_w_volatility + eff_w_volume
    return eff_w_trend, eff_w_momentum, eff_w_volatility, eff_w_volume, total_weight


# 3.7 Composite Oscillator
def calculate_oscillator(eff_w_trend, eff_w_momentum, eff_w_volatility, eff_w_volume, norm_trend, momentum,
                         norm_volatility, norm_volume, total_weight):
    """Calculates composite oscillator."""
    oscillator_raw = (eff_w_trend * norm_trend + eff_w_momentum * momentum +
                      eff_w_volatility * norm_volatility + eff_w_volume * norm_volume) / total_weight
    oscillator = oscillator_raw * 100
    return f_nr(oscillator, True)


# 3.8 Dynamic Thresholds
def calculate_thresholds(oscillator):
    """Calculates dynamic upper and lower thresholds."""
    osc_std = oscillator.rolling(window=config.oscLookback).std()
    upper_threshold = config.baseUpper + (osc_std * config.dynMultiplier)
    lower_threshold = config.baseLower - (osc_std * config.dynMultiplier)
    return f_nr(upper_threshold, True), f_nr(lower_threshold, True)


# 3.9 AI Forecast (Optional)
def calculate_ai_forecast(close, oscillator):
    """Calculates AI forecast and bias."""
    if config.useAI:
        # Linear regression for forecasting
        def linreg(series, length, offset):
            x = np.arange(length)
            y = series[-length:] if offset == 0 else series[-length - offset:-offset]
            if len(y) < length:
                return np.nan
            coeffs = np.polyfit(x, y, 1)
            return np.polyval(coeffs, length - 1)

        ai_value = close.rolling(window=config.ai_length).apply(lambda x: linreg(x, config.ai_length, 0), raw=True)
        ai_slope = ai_value - close.rolling(window=config.ai_length).apply(lambda x: linreg(x, config.ai_length, 1),
                                                                           raw=True)
        ai_pred = ai_value + ai_slope * config.ai_future
        ai_pred_smooth = ai_pred.rolling(window=2).mean() if config.aiSmoothing else ai_pred

        osc_lb = oscillator.rolling(window=config.ai_length).min()
        osc_hb = oscillator.rolling(window=config.ai_length).max()
        ai_lb = ai_pred_smooth.rolling(window=config.ai_length).min()
        ai_hb = ai_pred_smooth.rolling(window=config.ai_length).max()
        osc_range = np.maximum(osc_hb - osc_lb, 0.001)
        ai_range = np.maximum(ai_hb - ai_lb, 0.001)
        ai_osc = ((ai_pred_smooth - ai_lb) / ai_range) * osc_range + osc_lb
        osc_mid = (config.baseUpper + config.baseLower) / 2
        ai_bias = ai_osc - osc_mid
        return f_nr(ai_bias, True), ai_osc
    return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)


# 3.10 Trend via Local HMA
def calculate_trend_hma(close):
    """Calculates trend based on HMA."""
    local_hma = hma(close, config.hmaLen)
    trend_hma = np.where(close > local_hma, 1.0, np.where(close < local_hma, -1.0, 0.0))
    return f_nr(pd.Series(trend_hma, index=close.index), True)


# 3.11 Higher Timeframe Trend (if enabled)
def calculate_htf_trend(close, htf_data):
    """Calculates trend for higher timeframe."""
    if config.useHTF:
        htf_hma = hma(htf_data['close'], config.hmaLen)
        htf_trend = np.where(close > htf_hma, 1.0, np.where(close < htf_hma, -1.0, 0.0))
        return f_nr(pd.Series(htf_trend, index=close.index), True)
    return pd.Series(0.0, index=close.index)


# ─────────────────────────────────────────────
# 4. SIGNAL GENERATION
# Purpose: Generate buy/sell signals
# ─────────────────────────────────────────────
def generate_signals(oscillator, upper_threshold, lower_threshold, trend_hma, htf_trend, adx_val):
    """Generates immediate and latched signals."""
    # Immediate Signals
    cond_strong_buy = (oscillator > upper_threshold) & (trend_hma > 0) & ((not config.useHTF) | (htf_trend > 0)) & (
                adx_val > 20)
    cond_strong_sell = (oscillator < lower_threshold) & (trend_hma < 0) & ((not config.useHTF) | (htf_trend < 0)) & (
                adx_val > 20)
    cond_early_buy = (oscillator > config.baseUpper) & (oscillator < upper_threshold)
    cond_early_sell = (oscillator < config.baseLower) & (oscillator > lower_threshold)

    immediate_signal = pd.Series("No Signal", index=oscillator.index)
    immediate_signal = np.where(cond_strong_buy, "Strong Buy", immediate_signal)
    immediate_signal = np.where(cond_strong_sell, "Strong Sell", immediate_signal)
    immediate_signal = np.where(cond_early_buy, "Early Buy", immediate_signal)
    immediate_signal = np.where(cond_early_sell, "Early Sell", immediate_signal)

    # Latched Signals
    stored_state = pd.Series("Neutral", index=oscillator.index)
    for i in range(1, len(oscillator)):
        if stored_state.iloc[i - 1] == "Neutral":
            if cond_early_buy.iloc[i]:
                stored_state.iloc[i] = "EarlyBuy"
            elif cond_early_sell.iloc[i]:
                stored_state.iloc[i] = "EarlySell"
            elif cond_strong_buy.iloc[i]:
                stored_state.iloc[i] = "Long"
            elif cond_strong_sell.iloc[i]:
                stored_state.iloc[i] = "Short"
            else:
                stored_state.iloc[i] = "Neutral"
        elif stored_state.iloc[i - 1] == "EarlyBuy":
            if cond_strong_buy.iloc[i]:
                stored_state.iloc[i] = "Long"
            elif not cond_early_buy.iloc[i]:
                stored_state.iloc[i] = "Neutral"
            else:
                stored_state.iloc[i] = "EarlyBuy"
        elif stored_state.iloc[i - 1] == "Long":
            if cond_strong_sell.iloc[i]:
                stored_state.iloc[i] = "Short"
            elif cond_early_sell.iloc[i]:
                stored_state.iloc[i] = "EarlySell"
            else:
                stored_state.iloc[i] = "Long"
        elif stored_state.iloc[i - 1] == "EarlySell":
            if cond_strong_sell.iloc[i]:
                stored_state.iloc[i] = "Short"
            elif not cond_early_sell.iloc[i]:
                stored_state.iloc[i] = "Neutral"
            else:
                stored_state.iloc[i] = "EarlySell"
        elif stored_state.iloc[i - 1] == "Short":
            if cond_strong_buy.iloc[i]:
                stored_state.iloc[i] = "Long"
            elif cond_early_buy.iloc[i]:
                stored_state.iloc[i] = "EarlyBuy"
            else:
                stored_state.iloc[i] = "Short"

    stored_signal = np.where(stored_state == "Long", "Strong Buy",
                             np.where(stored_state == "Short", "Strong Sell",
                                      np.where(stored_state == "EarlyBuy", "Early Buy",
                                               np.where(stored_state == "EarlySell", "Early Sell", "No Signal"))))

    return immediate_signal, stored_signal


# ─────────────────────────────────────────────
# 5. MAIN FUNCTION
# Purpose: Orchestrate all calculations and return results
# ─────────────────────────────────────────────
def pro_scalper_ai(df, htf_data=None):
    """Main function to run Pro Scalper AI."""
    # Prepare confirmed data
    cOpen, cHigh, cLow, cClose, cVolume = prepare_data(df)

    # Calculate indicators
    confirmed_adx, confirmed_norm_trend = calculate_adx_di(cHigh, cLow, cClose, config.adxLen, config.diLen)
    confirmed_norm_volatility = calculate_atr(cHigh, cLow, cClose, config.atrLen)
    confirmed_momentum = f_nr(calculate_momentum(cClose, cHigh, cLow, config.momChoice), True)
    confirmed_tdi_rsi, confirmed_tdi_signal = calculate_tdi(cClose, config.tdiRsiLen, config.tdiSignalLen)
    confirmed_volume_norm = calculate_obv(cClose, cVolume)

    # Dynamic weighting
    norm_atr = (AverageTrueRange(high=cHigh, low=cLow, close=cClose,
                                 window=config.atrLen).average_true_range() / cClose)
    eff_w_trend, eff_w_momentum, eff_w_volatility, eff_w_volume, total_weight = calculate_dynamic_weights(confirmed_adx,
                                                                                                          norm_atr)

    # Composite oscillator
    oscillator = calculate_oscillator(eff_w_trend, eff_w_momentum, eff_w_volatility, eff_w_volume,
                                      confirmed_norm_trend, confirmed_momentum, confirmed_norm_volatility,
                                      confirmed_volume_norm, total_weight)

    # Dynamic thresholds
    upper_threshold, lower_threshold = calculate_thresholds(oscillator)

    # AI forecast
    ai_bias, ai_osc = calculate_ai_forecast(cClose, oscillator)
    adjusted_upper_threshold = upper_threshold + (
                ai_bias > 0) * config.aiAdjustment if config.useAI else upper_threshold
    adjusted_lower_threshold = lower_threshold - (
                ai_bias < 0) * config.aiAdjustment if config.useAI else lower_threshold

    # Trend calculations
    confirmed_trend_hma = calculate_trend_hma(cClose)
    confirmed_htf_trend = calculate_htf_trend(cClose, htf_data) if htf_data is not None else pd.Series(0.0,
                                                                                                       index=cClose.index)

    # Generate signals
    immediate_signal, stored_signal = generate_signals(oscillator, adjusted_upper_threshold, adjusted_lower_threshold,
                                                       confirmed_trend_hma, confirmed_htf_trend, confirmed_adx)

    # Compile results
    results = pd.DataFrame({
         'open': cOpen,
        'high': cHigh,
        'low': cLow,
        'close': cClose,
        'volume': cVolume,
        'oscillator': oscillator,
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold,
        'adjusted_upper_threshold': adjusted_upper_threshold,
        'adjusted_lower_threshold': adjusted_lower_threshold,
        'adx': confirmed_adx,
        'trend_hma': confirmed_trend_hma,
        'htf_trend': confirmed_htf_trend,
        'momentum': confirmed_momentum,
        'volume_norm': confirmed_volume_norm,
        'ai_bias': ai_bias,
        'immediate_signal': immediate_signal,
        'stored_signal': stored_signal
    }, index=df.index)

    return results

# Example usage (uncomment to test)
if __name__ == "__main__":
    # Sample data (replace with actual OHLCV data)
    df = pd.DataFrame({
        'open': np.random.rand(100),
        'high': np.random.rand(100),
        'low': np.random.rand(100),
        'close': np.random.rand(100),
        'volume': np.random.rand(100) * 1000
    })
    results = pro_scalper_ai(df)
    print(results.tail())