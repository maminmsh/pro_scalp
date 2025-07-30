import ccxt
import ccxt.pro
import asyncio
from datetime import datetime, timedelta

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