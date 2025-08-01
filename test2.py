import uuid
import pandas as pd
from main_with_sqlite import load_from_sqlite, save_to_sqlite
from datetime import datetime
import jdatetime


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
        signal = row['signal']
        entry_price = row['price']
        price = row['price']

        # Combine jalali_date and jalali_time for timestamp
        timestamp = f"{row['jalali_date']} {row['jalali_time']}"

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
                    'exit_time': timestamp,
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
                'entry_time': timestamp
            }
            entry_price = price

            # Record trade entry
            trades.append({
                'trade_id': str(uuid.uuid4()),
                'entry_time': timestamp,
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