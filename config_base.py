# Configuration file for Pro Scalper AI parameters

# Higher Timeframe Trend Filter
useHTF = False                  # Use Higher Timeframe Trend Filter?
htf_tf = "15"                   # HTF timeframe (e.g., "15" for 15 minutes)

# MACD and Pinbar Filters
useMACDFilter = False           # Use MACD Filter?
usePinbarFilter = False         # Use Pinbar Filter?

# Indicator Lengths
adxLen = 14                     # ADX Length
diLen = 14                      # DI Length
atrLen = 14                     # ATR Length
hmaLen = 21                     # HMA Length

# Momentum Mode
momChoice = "Stochastic RSI"    # Momentum Mode: "Stochastic RSI", "Fisher RSI", or "Williams %R"

# Stochastic RSI Parameters
stochRsiLen = 12                # Stoch RSI Length
stochKLen = 3                   # Stoch RSI %K Smoothing
stochDLen = 3                   # Stoch RSI %D Smoothing

# Fisher RSI Parameters
fisherRsiLen = 14               # Fisher RSI Length

# Williams %R Parameters
wprLen = 14                     # Williams %R Length

# TDI Parameters
tdiRsiLen = 13                  # TDI RSI Length
tdiSignalLen = 7                # TDI Signal Length

# Weights and Thresholds
wTrend = 0.3                    # Trend Weight
wMomentum = 0.3                 # Momentum Weight
wVolatility = 0.2               # Volatility Weight
wVolume = 0.2                   # Volume Weight
baseUpper = 25.0                # Base Upper Threshold
baseLower = -25.0               # Base Lower Threshold
dynMultiplier = 0.5             # Dynamic Threshold Multiplier
oscLookback = 50                # Oscillator Std Dev Lookback

# Adaptive Hysteresis
useHysteresis = True            # Use Adaptive Hysteresis?
baseHystGap = 1.0               # Base Hysteresis Gap
hystMultiplier = 1.0            # Hysteresis Volatility Multiplier

# Dynamic Weighting
useDynamicW = True              # Use Dynamic Weighting?
maxVolFactor = 2.0              # Max Volatility Factor
maxTrendFactor = 1.5            # Max Trend Factor

# Volume Lookback
volLookback = 50                # Volume Lookback

# AI Forecast Parameters
useAI = True                   # Use AI Forecast?
# useAI = False                   # Use AI Forecast?
ai_length = 20                  # AI Length
ai_future = 2                   # AI Forecast Bars Ahead
aiAdjustment = 1.5              # AI Threshold Shift
aiSmoothing = True              # Smooth AI Forecast?

# Latching Mode
useLatching = True              # Enable Latching?

# Dashboard Text Size
dashboardSizeOption = "Normal"  # Dashboard Text Size: "Tiny", "Small", "Normal", "Large"