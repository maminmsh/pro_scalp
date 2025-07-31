import uuid

import pandas as pd
import sqlite3
from main_with_sqlite import load_from_sqlite, save_to_sqlite


# # تابع شبیه‌سازی معاملات
# def simulate_trading(df, initial_balance=200, leverage=10, fee_percent=0.02, profit_threshold=0.01,
#                      max_loss_percent=0.10):
#     """
#     شبیه‌سازی معاملات بر اساس سیگنال‌ها
#     :param df: DataFrame حاوی داده‌های سیگنال و قیمت
#     :param initial_balance: سرمایه اولیه (دلار)
#     :param leverage: اهرم معامله
#     :param fee_percent: درصد کارمزد هر معامله
#     :param profit_threshold: آستانه سود برای بستن پوزیشن
#     :param max_loss_percent: حداکثر ضرر مجاز (درصد از سرمایه اولیه)
#     :return: DataFrame معاملات
#     """
#     balance = initial_balance
#     max_loss = initial_balance * max_loss_percent
#     position = None
#     entry_price = 0
#     trade_volume = initial_balance * leverage
#     trades = []
#
#     for index, row in df.iterrows():
#         if balance <= 0 or (initial_balance - balance) > max_loss:
#             print("معاملات متوقف شد: سرمایه صفر یا ضرر بیش از حد")
#             break
#
#         signal = row['stored_signal']
#
#         # محاسبه سود/ضرر برای پوزیشن باز
#         if position in ['Long', 'Short']:
#             current_price = row['close']
#             if position == 'Long':
#                 profit_percent = (current_price - entry_price) / entry_price
#             else:  # Short
#                 profit_percent = (entry_price - current_price) / entry_price
#
#             # بستن پوزیشن در صورت سود بیش از 1%
#             if profit_percent >= profit_threshold:
#                 profit = profit_percent * trade_volume
#                 fee = trade_volume * fee_percent * 2
#                 net_profit = profit - fee
#                 balance += net_profit
#
#                 trade_data = {
#                     'timestamp': index,
#                     'signal': signal,
#                     'position': position,
#                     'entry_price': entry_price,
#                     'exit_price': current_price,
#                     'profit_percent': profit_percent * 100,
#                     'net_profit': net_profit,
#                     'balance': balance
#                 }
#                 trades.append(trade_data)
#                 position = None
#
#         # بررسی تغییر سیگنال
#         if signal in ['Strong Buy', 'Early Buy'] and position != 'Long':
#             if position:
#                 current_price = row['close']
#                 if position == 'Long':
#                     profit_percent = (current_price - entry_price) / entry_price
#                 else:
#                     profit_percent = (entry_price - current_price) / entry_price
#                 profit = profit_percent * trade_volume
#                 fee = trade_volume * fee_percent * 2
#                 net_profit = profit - fee
#                 balance += net_profit
#
#                 trade_data = {
#                     'timestamp': index,
#                     'signal': signal,
#                     'position': position,
#                     'entry_price': entry_price,
#                     'exit_price': current_price,
#                     'profit_percent': profit_percent * 100,
#                     'net_profit': net_profit,
#                     'balance': balance
#                 }
#                 trades.append(trade_data)
#
#             position = 'Long'
#             entry_price = row['close']
#             trade_volume = balance * leverage
#
#         elif signal in ['Strong Sell', 'Early Sell'] and position != 'Short':
#             if position:
#                 current_price = row['close']
#                 if position == 'Long':
#                     profit_percent = (current_price - entry_price) / entry_price
#                 else:
#                     profit_percent = (entry_price - current_price) / entry_price
#                 profit = profit_percent * trade_volume
#                 fee = trade_volume * fee_percent * 2
#                 net_profit = profit - fee
#                 balance += net_profit
#
#                 trade_data = {
#                     'timestamp': index,
#                     'signal': signal,
#                     'position': position,
#                     'entry_price': entry_price,
#                     'exit_price': current_price,
#                     'profit_percent': profit_percent * 100,
#                     'net_profit': net_profit,
#                     'balance': balance
#                 }
#                 trades.append(trade_data)
#
#             position = 'Short'
#             entry_price = row['close']
#             trade_volume = balance * leverage
#
#     trades_df = pd.DataFrame(trades)
#     if not trades_df.empty:
#         save_to_sqlite(trades_df, db_name="trading_data.db", table_name="trades")
#     return trades_df

def simulate_trading(signals_df, initial_balance=200, leverage=10, fee_percent=0.02, profit_threshold=0.01,
                     max_loss_percent=0.10):
    # Initialize variables
    balance = initial_balance
    trades = []
    position = None
    entry_price = 0
    trade_volume = 0

    # Calculate max allowable loss
    max_loss = initial_balance * max_loss_percent

    for index, row in signals_df.iterrows():
        print(row)
        signal = row['signal']
        price = row['price']

        # Stop trading if loss exceeds max_loss_percent
        if initial_balance - balance >= max_loss:
            break

        # Close existing position if signal changes or profit threshold met
        if position is not None:
            # Calculate current profit/loss
            if position['type'] in ['Strong Buy', 'Early Buy']:
                profit_percent = (price - entry_price) / entry_price
            else:  # Strong Sell, Early Sell
                profit_percent = (entry_price - price) / entry_price

            # Close position if profit > threshold or signal changes
            if (profit_percent >= profit_threshold or
                    signal != position['type'] or
                    signal == 'No Signal'):
                # Calculate trade outcome
                profit_loss = trade_volume * profit_percent
                fee = trade_volume * fee_percent
                balance += profit_loss - fee

                # Record trade
                trades.append({
                    'trade_id': str(uuid.uuid4()),
                    'entry_time': position['entry_time'],
                    'exit_time': row['timestamp'],
                    'type': position['type'],
                    'entry_price': entry_price,
                    'exit_price': price,
                    'volume': trade_volume,
                    'profit_percent': profit_percent * 100,
                    'profit_loss': profit_loss,
                    'fee': fee,
                    'balance': balance
                })
                position = None

        # Open new position if signal is not 'No Signal' and no open position
        if position is None and signal != 'No Signal' and balance > 0:
            trade_volume = balance * leverage
            fee = trade_volume * fee_percent
            balance -= fee

            position = {
                'type': signal,
                'entry_time': row['timestamp']
            }
            entry_price = price

            # Record trade entry
            trades.append({
                'trade_id': str(uuid.uuid4()),
                'entry_time': row['timestamp'],
                'exit_time': None,
                'type': signal,
                'entry_price': price,
                'exit_price': None,
                'volume': trade_volume,
                'profit_percent': 0,
                'profit_loss': 0,
                'fee': fee,
                'balance': balance
            })

    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)

    # Format the DataFrame
    if not trades_df.empty:
        trades_df = trades_df.round(2)
        trades_df['profit_percent'] = trades_df['profit_percent'].apply(lambda x: f"{x:.2f}%")
        trades_df = trades_df[['trade_id', 'entry_time', 'exit_time', 'type',
                               'entry_price', 'exit_price', 'volume',
                               'profit_percent', 'profit_loss', 'fee', 'balance']]

    return trades_df


# 📌 اجرای تست
if __name__ == "__main__":
    # results = load_from_sqlite(db_name="trading_data.db", table_name="signals")
    results = load_from_sqlite(db_name="trading_data.db", table_name="signal_changes")
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