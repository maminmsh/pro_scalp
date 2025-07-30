import asyncio
import pandas as pd
import get_data as gd
from app import pro_scalper_ai
from load_save_data import save_to_sqlite, load_from_sqlite

# 📌 اجرای نمونه‌ها:
if __name__ == "__main__":
    # گرفتن 1000 کندل یک‌دقیقه‌ای آخر
    candles = gd.get_historical_ohlcv(limit=1000)
    print("نمونه کندل:", candles[-1])  # آخرین کندل

    # گرفتن کندل از بازه تاریخی مشخص
    candles_range = gd.get_historical_ohlcv(since="2024-07-01T00:00:00Z", to="2024-07-02T00:00:00Z")
    print(f"{len(candles_range)} کندل از بازه مشخص دریافت شد")

    # تبدیل داده‌های واقعی به DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # حذف ستون‌های غیرضروری (مثل timestamp) اگر وجود داشته باشند
    df = df[['timestamp','open', 'high', 'low', 'close', 'volume']]

    # اطمینان از اینکه داده‌ها به نوع عددی تبدیل شده‌اند
    df = df.astype({
        'open': float,
        'high': float,
        'low': float,
        'close': float,
        'volume': float
    })

    # اجرای تابع pro_scalper_ai با داده‌های واقعی
    results = pro_scalper_ai(df)
    print("نتایج Pro Scalper AI:")
    print(results.tail())

    # Strong Buy
    # Strong Sell
    # Early Buy
    # Early Sell
    # No Signal

    # ذخیره نتایج در SQLite
    save_to_sqlite(results, db_name="trading_data.db", table_name="signals")

    # بازیابی داده‌ها برای بررسی
    loaded_df = load_from_sqlite(db_name="trading_data.db", table_name="signals")
    print("داده‌های بازیابی‌شده از SQLite:")
    print(loaded_df.tail())

    # اجرای WebSocket به صورت async (همیشه در انتهای فایل)
    asyncio.run(gd.watch_ticker())