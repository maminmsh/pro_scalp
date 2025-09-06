import ccxt
import ccxt.pro
import asyncio
from datetime import datetime, timedelta

import pandas as pd


# تابع اول: گرفتن کندل‌ها
def get_historical_ohlcv(symbol='BTC/USDT', timeframe='5m', limit=1000, since=None, to=None):
    exchange = ccxt.binance()
    exchange.load_markets()

    if since:
        # تبدیل زمان به timestamp
        since_ts = exchange.parse8601(since)
        candles = []
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
            if not ohlcv:
                break
            candles += ohlcv
            since_ts = ohlcv[-1][0] + 1
            if to and since_ts >= exchange.parse8601(to):
                break
            if len(ohlcv) < limit:
                break
        return candles
    else:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

# تابع دوم: گرفتن قیمت لحظه‌ای از طریق WebSocket
async def watch_ticker(symbol='BTC/USDT'):
    exchange = ccxt.pro.binance()
    try:
        while True:
            ticker = await exchange.watch_ticker(symbol)
            print(f"[{ticker['datetime']}] {symbol} price: {ticker['last']} USDT")
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        await exchange.close()


def get_historical_ohlcv_df(symbol='BTC/USDT', timeframe='5m', limit=1000):
    exchange = ccxt.binance()
    exchange.load_markets()

    # گرفتن داده‌ها
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # ساخت DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # تبدیل timestamp از میلی‌ثانیه به datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # تنظیم ایندکس به timestamp
    df.set_index('timestamp', inplace=True)

    # انتخاب ستون‌های مورد نیاز و تبدیل به float
    df_ohlcv = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    return df_ohlcv


# def get_historical_ohlcv_df(symbol='BTC/USDT', timeframe='5m', limit=1000):
#     exchange = ccxt.binance()
#     exchange.load_markets()
#
#     candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
#     df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#
#     # تنظیم ایندکس به timestamp
#     df.set_index('timestamp', inplace=True)
#
#     # حذف ستون‌های غیرضروری
#     df_ohlcv = df[['open', 'high', 'low', 'close', 'volume']]
#
#     # اطمینان از اینکه داده‌ها به نوع عددی تبدیل شده‌اند
#     df_ohlcv = df_ohlcv.astype({
#         'open': float,
#         'high': float,
#         'low': float,
#         'close': float,
#         'volume': float
#     })
#     df_ohlcv =ensure_datetime_index(df_ohlcv)
#     return df_ohlcv
#
#
# def ensure_datetime_index(df):
#     if not isinstance(df.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
#         if 'time' in df.columns:
#             df['time'] = pd.to_datetime(df['time'])
#             df = df.set_index('time')
#         elif 'date' in df.columns:
#             df['date'] = pd.to_datetime(df['date'])
#             df = df.set_index('date')
#         else:
#             raise ValueError("No datetime column found to set as index")
#     return df