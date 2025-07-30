import pandas as pd
import sqlite3
from main_with_sqlite import load_from_sqlite, save_to_sqlite


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
    max_loss = initial_balance * max_loss_percent
    position = None
    entry_price = 0
    trade_volume = initial_balance * leverage
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
                position = None

        # بررسی تغییر سیگنال
        if signal in ['Strong Buy', 'Early Buy'] and position != 'Long':
            if position:
                current_price = row['close']
                if position == 'Long':
                    profit_percent = (current_price - entry_price) / entry_price
                else:
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

            position = 'Long'
            entry_price = row['close']
            trade_volume = balance * leverage

        elif signal in ['Strong Sell', 'Early Sell'] and position != 'Short':
            if position:
                current_price = row['close']
                if position == 'Long':
                    profit_percent = (current_price - entry_price) / entry_price
                else:
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

            position = 'Short'
            entry_price = row['close']
            trade_volume = balance * leverage

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        save_to_sqlite(trades_df, db_name="trading_data.db", table_name="trades")
    return trades_df


# 📌 اجرای تست
if __name__ == "__main__":
    results = load_from_sqlite(db_name="trading_data.db", table_name="signals")
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