import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
import get_data as gd
from main_with_sqlite import add_jalali_datetime, save_to_sqlite, save_signal_changes


# ────────────────────────────────
# تابع سوپرترند
# ────────────────────────────────
def supertrend(df, factor=3, atr=None):
    hl2 = (df['High'] + df['Low']) / 2
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)

    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()

    for i in range(1, len(df)):
        if (upperband.iloc[i] < final_upperband.iloc[i-1]) or (df['Close'].iloc[i-1] > final_upperband.iloc[i-1]):
            final_upperband.iloc[i] = upperband.iloc[i]
        else:
            final_upperband.iloc[i] = final_upperband.iloc[i-1]

        if (lowerband.iloc[i] > final_lowerband.iloc[i-1]) or (df['Close'].iloc[i-1] < final_lowerband.iloc[i-1]):
            final_lowerband.iloc[i] = lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = final_lowerband.iloc[i-1]

    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if supertrend.iloc[i-1] == final_upperband.iloc[i-1]:
            if df['Close'].iloc[i] > upperband.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1
        else:
            if df['Close'].iloc[i] < lowerband.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1

        supertrend.iloc[i] = (
            final_lowerband.iloc[i] if direction.iloc[i] == -1 else final_upperband.iloc[i]
        )

    return supertrend, direction

# ────────────────────────────────
# خوشه‌بندی سه‌تایی روی ATR
# ────────────────────────────────
def kmeans_atr(volatility, train_len=100, high_guess=0.75, mid_guess=0.5, low_guess=0.25, max_iter=50):
    vol_window = volatility[-train_len:]

    vmin, vmax = vol_window.min(), vol_window.max()
    hv = vmin + (vmax - vmin) * high_guess
    mv = vmin + (vmax - vmin) * mid_guess
    lv = vmin + (vmax - vmin) * low_guess

    centroids = np.array([hv, mv, lv])

    for _ in range(max_iter):
        clusters = {0: [], 1: [], 2: []}
        for v in vol_window:
            idx = np.argmin(np.abs(v - centroids))
            clusters[idx].append(v)
        new_centroids = np.array([np.mean(clusters[k]) if clusters[k] else centroids[k] for k in range(3)])
        if np.allclose(new_centroids, centroids, rtol=1e-5, atol=1e-5):
            break
        centroids = new_centroids

    return centroids

# ────────────────────────────────
# تابع اصلی Adaptive SuperTrend + سیگنال‌ها + تاریخ شمسی
# ────────────────────────────────
def adaptive_supertrend(df, atr_len=10, factor=3, train_len=100):
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=atr_len).average_true_range()
    df['ATR'] = atr

    supertrend_vals = []
    dir_vals = []
    cluster_vals = []
    signals = []

    for i in range(len(df)):
        if i < train_len:
            supertrend_vals.append(np.nan)
            dir_vals.append(np.nan)
            cluster_vals.append(np.nan)
            signals.append("No Signal")
            continue

        centroids = kmeans_atr(atr[:i+1], train_len=train_len)
        dists = np.abs(atr.iloc[i] - centroids)
        cluster = np.argmin(dists)
        chosen_atr = centroids[cluster]

        st, direction = supertrend(df.iloc[:i+1], factor=factor, atr=atr.iloc[:i+1].replace(atr.iloc[i], chosen_atr))
        supertrend_vals.append(st.iloc[-1])
        dir_vals.append(direction.iloc[-1])
        cluster_vals.append(cluster)

        # تعریف سیگنال
        if direction.iloc[-1] == 1 and cluster == 2:
            signal = "Strong Buy"
        elif direction.iloc[-1] == 1 and cluster == 1:
            signal = "Early Buy"
        elif direction.iloc[-1] == -1 and cluster == 2:
            signal = "Strong Sell"
        elif direction.iloc[-1] == -1 and cluster == 1:
            signal = "Early Sell"
        else:
            signal = "No Signal"

        signals.append(signal)

    df['Adaptive_ST'] = supertrend_vals
    df['Direction'] = dir_vals
    df['Cluster'] = cluster_vals
    df['stored_signal'] = signals

    # تبدیل تاریخ میلادی به جلالی
    return df

# ────────────────────────────────
# تست روی دیتا
# ────────────────────────────────
if __name__ == "__main__":
    candles = gd.get_historical_ohlcv(limit=1000,timeframe='5m')
    print("نمونه کندل:", candles[-1])


    # تبدیل داده‌های واقعی به DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

    # تنظیم ایندکس به timestamp
    df.set_index('timestamp', inplace=True)

    df = adaptive_supertrend(df)
    df.columns = df.columns.str.lower()

    results = add_jalali_datetime(df)

    save_to_sqlite(results, db_name="trading_data.db", table_name="signals")
    save_signal_changes(results, db_name="trading_data.db", table_name="signal_changes")


    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    print(df.tail(20))
