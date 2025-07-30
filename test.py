import pandas as pd
import sqlite3

from main_with_sqlite import load_from_sqlite, save_to_sqlite


# تابع برای ذخیره معاملات در SQLite
def save_trade_to_sqlite(trade_data, db_name="trading_data.db", table_name="trades"):
    """
    ذخیره اطلاعات معامله در جدول trades
    :param trade_data: دیکشنری حاوی اطلاعات معامله
    :param db_name: نام فایل پایگاه داده
    :param table_name: نام جدول
    """
    conn = sqlite3.connect(db_name)
    df_trade = pd.DataFrame([trade_data])
    df_trade.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()
    print(f"معامله در جدول '{table_name}' ذخیره شد.")


# تابع شبیه‌سازی معاملات
def simulate_trading(df, initial_balance=200, leverage=10, fee_percent=0.02, profit_threshold=0.01,
                     max_loss_percent=0.10):
    """
    شبیه‌سازی معاملات بر اساس سیگنال‌ها
    :param df: DataFrame حاوی داده‌های سیگنال و قیمت
    :param initial_balance: سرمایه اولیه (دلار)
    :param leverage: اهرم معامله
    :param fee_percent: درصد کارمزد هر معامله
    :param profit_threshold: آستانه سود برای بستن پوزیشن
    :param max_loss_percent: حداکثر ضرر مجاز (درصد از سرمایه اولیه)
    :return: DataFrame معاملات
    """
    balance = initial_balance
    max_loss = initial_balance * max_loss_percent  # حداکثر ضرر (20 دلار)
    position = None  # None, 'Long', 'Short'
    entry_price = 0
    trade_volume = initial_balance * leverage  # حجم معامله با اهرم
    trades = []

    for index, row in df.iterrows():
        if balance <= 0 or (initial_balance - balance) > max_loss:
            print("معاملات متوقف شد: سرمایه صفر یا ضرر بیش از حد")
            break

        signal = row['stored_signal']

        # محاسبه سود/ضرر برای پوزیشن باز
        if position in ['Long', 'Short']:
            current_price = row['close']
            if position == 'Long':
                profit_percent = (current_price - entry_price) / entry_price
            else:  # Short
                profit_percent = (entry_price - current_price) / entry_price

            # بستن پوزیشن در صورت سود بیش از 1%
            if profit_percent >= profit_threshold:
                profit = profit_percent * trade_volume
                fee = trade_volume * fee_percent * 2  # کارمزد ورود و خروج
                net_profit = profit - fee
                balance += net_profit

                trade_data = {
                    'timestamp': index,
                    'signal': signal,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'profit_percent': profit_percent * 100,
                    'net_profit': net_profit,
                    'balance': balance
                }
                trades.append(trade_data)
                save_trade_to_sqlite(trade_data)
                position = None

        # بررسی تغییر سیگنال
        if signal in ['Strong Buy', 'Early Buy'] and position != 'Long':
            if position:  # بستن پوزیشن قبلی
                current_price = row['close']
                if position == 'Long':
                    profit_percent = (current_price - entry_price) / entry_price
                else:  # Short
                    profit_percent = (entry_price - current_price) / entry_price
                profit = profit_percent * trade_volume
                fee = trade_volume * fee_percent * 2
                net_profit = profit - fee
                balance += net_profit

                trade_data = {
                    'timestamp': index,
                    'signal': signal,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'profit_percent': profit_percent * 100,
                    'net_profit': net_profit,
                    'balance': balance
                }
                trades.append(trade_data)
                save_trade_to_sqlite(trade_data)

            # باز کردن پوزیشن جدید
            position = 'Long'
            entry_price = row['close']
            trade_volume = balance * leverage  # به‌روزرسانی حجم معامله

        elif signal in ['Strong Sell', 'Early Sell'] and position != 'Short':
            if position:  # بستن پوزیشن قبلی
                current_price = row['close']
                if position == 'Long':
                    profit_percent = (current_price - entry_price) / entry_price
                else:  # Short
                    profit_percent = (entry_price - current_price) / entry_price
                profit = profit_percent * trade_volume
                fee = trade_volume * fee_percent * 2
                net_profit = profit - fee
                balance += net_profit

                trade_data = {
                    'timestamp': index,
                    'signal': signal,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'profit_percent': profit_percent * 100,
                    'net_profit': net_profit,
                    'balance': balance
                }
                trades.append(trade_data)
                save_trade_to_sqlite(trade_data)

            # باز کردن پوزیشن جدید
            position = 'Short'
            entry_price = row['close']
            trade_volume = balance * leverage

    return pd.DataFrame(trades)


# 📌 اجرای تست
if __name__ == "__main__":
    # بازیابی داده‌ها از SQLite
    results = load_from_sqlite(db_name="trading_data.db", table_name="signals")

    # شبیه‌سازی معاملات
    trades_df = simulate_trading(
        results,
        initial_balance=200,
        leverage=10,
        fee_percent=0.02,
        profit_threshold=0.01,
        max_loss_percent=0.10
    )

    print("نتایج معاملات:")
    print(trades_df)

    # ذخیره جدول معاملات در SQLite
    save_to_sqlite(trades_df, db_name="trading_data.db", table_name="trades")