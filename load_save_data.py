import asyncio
import pandas as pd
import sqlite3
import get_data as gd
from pro_scalp import pro_scalper_ai


# تابع برای ذخیره DataFrame در SQLite
def save_to_sqlite(df, db_name="trading_data.db", table_name="signals"):
    """
    ذخیره DataFrame در پایگاه داده SQLite
    :param df: DataFrame حاوی داده‌های سیگنال‌ها
    :param db_name: نام فایل پایگاه داده
    :param table_name: نام جدول
    """
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=True, index_label='timestamp')
    conn.close()
    print(f"داده‌ها در جدول '{table_name}' از پایگاه داده '{db_name}' ذخیره شدند.")


# تابع برای بازیابی داده‌ها از SQLite
def load_from_sqlite(db_name="trading_data.db", table_name="signals"):
    """
    بازیابی DataFrame از پایگاه داده SQLite
    :param db_name: نام فایل پایگاه داده
    :param table_name: نام جدول
    :return: DataFrame
    """
    conn = sqlite3.connect(db_name)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn, index_col='timestamp')
    conn.close()
    return df


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

    # تنظیم ایندکس به timestamp
    df.set_index('timestamp', inplace=True)

    # حذف ستون‌های غیرضروری
    df_ohlcv = df[['open', 'high', 'low', 'close', 'volume']]

    # اطمینان از اینکه داده‌ها به نوع عددی تبدیل شده‌اند
    df_ohlcv = df_ohlcv.astype({
        'open': float,
        'high': float,
        'low': float,
        'close': float,
        'volume': float
    })

    # پر کردن مقادیر گم‌شده (در صورت وجود)
    df_ohlcv = df_ohlcv.fillna(method='ffill')

    # اجرای تابع pro_scalper_ai با داده‌های واقعی
    results = pro_scalper_ai(df_ohlcv)
    print("نتایج Pro Scalper AI:")
    print(results.tail())

    # ذخیره نتایج در SQLite
    save_to_sqlite(results, db_name="trading_data.db", table_name="signals")

    # بازیابی داده‌ها برای بررسی
    loaded_df = load_from_sqlite(db_name="trading_data.db", table_name="signals")
    print("داده‌های بازیابی‌شده از SQLite:")
    print(loaded_df.tail())

    # اجرای WebSocket به صورت async (همیشه در انتهای فایل)
    asyncio.run(gd.watch_ticker())