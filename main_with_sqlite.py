import asyncio
import time

import ccxt.pro
import pandas as pd
import sqlite3
import jdatetime
import get_data as gd
from pro_scalp import pro_scalper_ai
import requests

TELEGRAM_TOKEN = "7404584711:AAGdB5l-ekkYmxOibWh0-D6t7CCLZptJZv4"
CHAT_ID = -1003023688443


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print("Telegram error:", e)

def add_jalali_datetime(df):
    """
    اضافه کردن ستون‌های تاریخ و ساعت جلالی به DataFrame با ایندکس datetime
    """
    df = df.copy()

    # اطمینان از اینکه ایندکس از نوع datetime میلادی است
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, unit='ms')  # ← یا unit='ms' اگر داده‌ها به میلی‌ثانیه هستند

    df['jalali_date'] = df.index.map(
        lambda x: jdatetime.datetime.fromgregorian(datetime=x).strftime('%Y/%m/%d')
    )
    df['jalali_time'] = df.index.map(
        lambda x: jdatetime.datetime.fromgregorian(datetime=x).strftime('%H:%M:%S')
    )
    return df


# تابع برای ذخیره DataFrame در SQLite
def save_to_sqlite(df, db_name="trading_data.db", table_name="signals"):
    """
    ذخیره DataFrame در پایگاه داده SQLite
    :param df: DataFrame
    :param db_name: نام فایل پایگاه داده
    :param table_name: نام جدول
    """
    if df.empty:
        print(f"هشدار: DataFrame برای جدول '{table_name}' خالی است.")
        return
    conn = sqlite3.connect(db_name)
    # برای جدول trades، ایندکس ذخیره نمی‌شود چون ستون timestamp در داده‌ها وجود دارد
    if table_name == "trades":
        df.to_sql(table_name, conn, if_exists='append', index=False)
    else:
        df.to_sql(table_name, conn, if_exists='replace', index=True, index_label='timestamp')
    conn.close()
    print(f"داده‌ها در جدول '{table_name}' از پایگاه داده '{db_name}' ذخیره شدند.")
# def save_to_sqlite(df, db_name="trading_data.db", table_name="signals"):
#     conn = sqlite3.connect(db_name)
#     df.to_sql(table_name, conn, if_exists='replace', index=True, index_label='timestamp')
#     conn.close()
#     print(f"داده‌ها در جدول '{table_name}' از پایگاه داده '{db_name}' ذخیره شدند.")


# تابع برای ذخیره تغییرات سیگنال در جدول جداگانه
# def save_signal_changes(df, db_name="trading_data.db", table_name="signal_changes"):
#     """
#     ذخیره تغییرات سیگنال در جدول جداگانه
#     :param df: DataFrame حاوی سیگنال‌ها
#     :param db_name: نام فایل پایگاه داده
#     :param table_name: نام جدول تغییرات سیگنال
#     """
#     # تشخیص تغییرات سیگنال
#     df['signal_change'] = df['stored_signal'].ne(df['stored_signal'].shift(1))
#     signal_changes = df[df['signal_change']][['stored_signal', 'close']]
#     signal_changes = signal_changes.rename(columns={'stored_signal': 'signal', 'close': 'price'})
#
#     # اضافه کردن تاریخ و ساعت جلالی
#     signal_changes = add_jalali_datetime(signal_changes)
#
#     # ذخیره در جدول
#     conn = sqlite3.connect(db_name)
#     signal_changes[['signal', 'price', 'jalali_date', 'jalali_time']].to_sql(
#         table_name, conn, if_exists='append', index=True, index_label='timestamp'
#     )
#     conn.close()
#     print(f"تغییرات سیگنال در جدول '{table_name}' ذخیره شدند.")


def save_signal_changes(df, db_name="trading_data.db", table_name="signal_changes"):
    """
    ذخیره تغییرات سیگنال در جدول جداگانه
    :param df: DataFrame حاوی سیگنال‌ها
    :param db_name: نام فایل پایگاه داده
    :param table_name: نام جدول تغییرات سیگنال
    """
    # تشخیص تغییرات سیگنال
    df['signal_change'] = df['stored_signal'].ne(df['stored_signal'].shift(1))

    # فقط سیگنال‌هایی که تغییر کردند و "No Signal" نیستند
    signal_changes = df[df['signal_change'] & (df['stored_signal'] != "No Signal")][['stored_signal', 'close']]
    signal_changes = signal_changes.rename(columns={'stored_signal': 'signal', 'close': 'price'})

    # درصد تغییرات قیمت نسبت به ردیف قبلی
    signal_changes['price_pct_change'] = signal_changes['price'].pct_change() * 100
    signal_changes['price_pct_change'] = signal_changes['price_pct_change'].fillna(0)

    # تعیین درست یا غلط بودن سیگنال نسبت به نوع آن
    def check_signal(row, prev_price):
        if prev_price is None:
            return 1  # اولین ردیف را درست فرض می‌کنیم
        if row['signal'] in ["Strong Buy", "Early Buy"]:
            return 1 if row['price'] > prev_price else 0
        elif row['signal'] in ["Strong Sell", "Early Sell"]:
            return 1 if row['price'] < prev_price else 0
        return 0

    prev_price = None
    correct_list = []
    for idx, row in signal_changes.iterrows():
        correct_list.append(check_signal(row, prev_price))
        prev_price = row['price']
    signal_changes['correct'] = correct_list

    # اضافه کردن تاریخ و ساعت جلالی
    signal_changes = add_jalali_datetime(signal_changes)

    # ذخیره در جدول
    conn = sqlite3.connect(db_name)
    signal_changes[['signal', 'price', 'price_pct_change', 'correct', 'jalali_date', 'jalali_time']].to_sql(
        table_name, conn, if_exists='append', index=True, index_label='timestamp'
    )
    conn.close()
    print(f"تغییرات سیگنال در جدول '{table_name}' ذخیره شدند.")


# تابع برای بازیابی داده‌ها از SQLite
def load_from_sqlite(db_name="trading_data.db", table_name="signals"):
    conn = sqlite3.connect(db_name)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn, index_col='timestamp')
    conn.close()
    return df



# 📌 اجرای نمونه‌ها:
# if __name__ == "__main__":
def main_with_sqlite():
    # گرفتن 1000 کندل 5m آخر
    candles = gd.get_historical_ohlcv(limit=1000,timeframe='5m')
    print("نمونه کندل:", candles[-1])

    # گرفتن کندل از بازه تاریخی مشخص
    # candles_range = gd.get_historical_ohlcv(since="2024-07-01T00:00:00Z", to="2024-07-02T00:00:00Z")
    # print(f"{len(candles_range)} کندل از بازه مشخص دریافت شد")

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
    # df_ohlcv = df_ohlcv.fillna(method='ffill')


    # اجرای تابع pro_scalper_ai
    results = pro_scalper_ai(df_ohlcv)
    print("نتایج Pro Scalper AI:")
    print(results.tail())

    # # اضافه کردن تاریخ و ساعت جلالی
    results = add_jalali_datetime(results)

    # # ذخیره نتایج در جدول signals
    save_to_sqlite(results, db_name="trading_data.db", table_name="signals")
    #
    # # ذخیره تغییرات سیگنال در جدول signal_changes
    save_signal_changes(results, db_name="trading_data.db", table_name="signal_changes")
    #
    # # بازیابی داده‌ها برای بررسی
    # loaded_df = load_from_sqlite(db_name="trading_data.db", table_name="signals")
    # print("داده‌های بازیابی‌شده از جدول signals:")
    # print(loaded_df.tail())
    #
    # loaded_signal_changes = load_from_sqlite(db_name="trading_data.db", table_name="signal_changes")
    # print("داده‌های بازیابی‌شده از جدول signal_changes:")
    # print(loaded_signal_changes.tail())

    # اجرای WebSocket به صورت async
    # asyncio.run(gd.watch_ticker())



if __name__ == "__main__":

    async def main_live_websocket(symbol='BTC/USDT'):
        exchange = ccxt.pro.binance()
        try:
            while True:
                try:
                    ticker = await exchange.watch_ticker(symbol)

                    # ساخت DataFrame از داده لحظه‌ای
                    df = pd.DataFrame([{
                        'timestamp': ticker['timestamp'],
                        'open': ticker['open'],
                        'high': ticker['high'],
                        'low': ticker['low'],
                        'close': ticker['last'],
                        'volume': ticker['baseVolume']
                    }])


                    df.set_index('timestamp', inplace=True)
                    df_ohlcv = df[['open','high','low','close','volume']].astype(float)


                    loaded_df = load_from_sqlite(db_name="trading_data.db", table_name="signals")
                    # print(df_ohlcv)
                    # print('sssssss')
                    # print(loaded_df)

                    # print('ssssssssssss')
                    # پردازش با الگوریتم Pro Scalper AI
                    # results = pro_scalper_ai(df_ohlcv)
                    results = pro_scalper_ai(loaded_df)
                    # print(results)
                    # results = add_jalali_datetime(results)

                    # print(results)

                    # ذخیره در SQLite
                    # save_to_sqlite(results, db_name="trading_data.db", table_name="signals")
                    # save_signal_changes(results, db_name="trading_data.db", table_name="signal_changes")

                    # ارسال آخرین سیگنال به تلگرام
                    send_telegram(f"📊 سیگنال جدید:\n{results.tail()}")

                except Exception as e:
                    send_telegram(f"❌ خطا در پردازش ticker: {str(e)}")
                    await asyncio.sleep(5)  # زمان کوتاه برای retry

        except Exception as e:
            send_telegram(f"❌ خطای WebSocket: {str(e)}")
        finally:
            await exchange.close()

    asyncio.run(main_live_websocket('BTC/USDT'))
