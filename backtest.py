import sqlite3
import pandas as pd


def load_from_sqlite(db_name="trading_data.db", table_name="signal_changes"):
    conn = sqlite3.connect(db_name)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


def backtest(capital, leverage, fee_rate, loss_limit, include_early,
             take_profit_pct, stop_loss_pct):
    df = load_from_sqlite()
    df.reset_index(drop=True, inplace=True)

    init_capital = capital
    open_trade = None
    cum_loss = 0

    long_signals = ["Strong Buy"]
    short_signals = ["Strong Sell"]

    if include_early:
        long_signals.append("Early Buy")
        short_signals.append("Early Sell")

    for i in range(1, len(df)):
        row = df.loc[i]
        signal, price, jdate, jtime = row["signal"], row["price"], row["jalali_date"], row["jalali_time"]

        if signal == "No Signal":
            continue

        if open_trade:
            entry_price = open_trade["price"]
            # درصد سود یا ضرر فعلی
            current_pnl_pct = (price - entry_price) / entry_price
            if open_trade["signal"] in short_signals:
                current_pnl_pct = -current_pnl_pct

            # شرط بسته شدن معامله:
            # 1. تغییر سیگنال
            # 2. رسیدن به حد سود
            # 3. رسیدن به حد ضرر
            if ((signal in long_signals and open_trade["signal"] in short_signals) or
                (signal in short_signals and open_trade["signal"] in long_signals)) or \
                    (current_pnl_pct >= take_profit_pct) or \
                    (current_pnl_pct <= -stop_loss_pct):

                exit_price = price
                volume = capital * leverage
                fee = volume * fee_rate * 2
                pnl = (exit_price - entry_price) / entry_price * volume
                if open_trade["signal"] in short_signals:
                    pnl = -pnl
                pnl -= fee
                capital += pnl
                cum_loss += pnl if pnl < 0 else 0
                icon = "🟢🟢🟢" if pnl > 0 else "🔴🔴🔴"

                print(f"""
                    📌 معامله بسته شد:
                    سیگنال: {open_trade['signal']} → {signal}
                    شروع: {entry_price:.2f} | پایان: {exit_price:.2f}
                    زمان ورود: {open_trade['jdate']} {open_trade['jtime']}
                    زمان خروج: {jdate} {jtime}
                     fee: {fee:.2f} USD
                    سود/ضرر: {pnl:.2f} USD ({pnl / init_capital * 100:.2f}%) 
                    سرمایه فعلی: {capital:.2f} USD
                    {icon} {i}
                """)

                # باز کردن معامله جدید فقط اگر سیگنال Buy یا Sell باشد
                if signal in long_signals + short_signals:
                    open_trade = {"signal": signal, "price": price, "jdate": jdate,
                                  "jtime": jtime, "timestamp": row["timestamp"]}
                else:
                    open_trade = None

                if abs(cum_loss) >= init_capital * loss_limit:
                    print("⛔ حد ضرر کل رسید. متوقف شد.")
                    break
        else:
            if signal in long_signals + short_signals:
                open_trade = {"signal": signal, "price": price,
                              "jdate": jdate, "jtime": jtime, "timestamp": row["timestamp"]}


# def backtest(capital, leverage, fee_rate, loss_limit, include_early=True):
#     df = load_from_sqlite()
#     df.reset_index(drop=True, inplace=True)
#
#     init_capital = capital
#     open_trade = None
#     cum_loss = 0
#
#     # دسته‌بندی سیگنال‌ها
#     long_signals = ["Strong Buy"]
#     short_signals = ["Strong Sell"]
#
#     if include_early:
#         long_signals.append("Early Buy")
#         short_signals.append("Early Sell")
#
#     for i in range(1, len(df)):
#         row = df.loc[i]
#         signal, price, jdate, jtime = row["signal"], row["price"], row["jalali_date"], row["jalali_time"]
#
#         # اگر سیگنال No Signal بود → رد شو
#         if signal == "No Signal":
#             continue
#
#         if open_trade:  # معامله باز داریم
#             # اگر نوع سیگنال عوض شد (مثلا buy → sell یا sell → buy)
#             if (signal in long_signals and open_trade["signal"] in short_signals) or \
#                (signal in short_signals and open_trade["signal"] in long_signals):
#
#                 # بستن معامله
#                 entry_price = open_trade["price"]
#                 exit_price = price
#                 volume = capital * leverage
#                 fee = volume * fee_rate * 2  # entry+exit
#                 pnl = (exit_price - entry_price) / entry_price * volume
#                 if open_trade["signal"] in short_signals:  # اگر شورت بود
#                     pnl = -pnl
#                 pnl -= fee
#                 capital += pnl
#                 cum_loss += pnl if pnl < 0 else 0
#                 icon = "🟢🟢🟢" if pnl > 0 else "🔴🔴🔴"
#
#                 print(f"""
#                     📌 معامله بسته شد:
#                     سیگنال: {open_trade['signal']} → {signal}
#                     شروع: {entry_price:.2f} | پایان: {exit_price:.2f}
#                     زمان ورود: {open_trade['jdate']} {open_trade['jtime']}
#                     زمان خروج: {jdate} {jtime}
#                     سود/ضرر: {pnl:.2f} USD ({pnl / init_capital * 100:.2f}%)
#                     سرمایه فعلی: {capital:.2f} USD
#                     {icon} {i}
#                 """)
#
#                 # باز کردن معامله جدید
#                 open_trade = {"signal": signal, "price": price, "jdate": jdate,
#                               "jtime": jtime, "timestamp": row["timestamp"]}
#
#                 if abs(cum_loss) >= init_capital * loss_limit:
#                     print("⛔ حد ضرر کل رسید. متوقف شد.")
#                     break
#         else:
#             # اولین معامله → فقط اگر Strong/Early Buy یا Sell بود
#             if signal in long_signals + short_signals:
#                 open_trade = {"signal": signal, "price": price,
#                               "jdate": jdate, "jtime": jtime, "timestamp": row["timestamp"]}

# if __name__ == "__main__":
def backtest_run():
    leverage = 2
    fee_rates = {1: 0.0005, 2: 0.001,3:0.0015, 5: 0.0025, 10: 0.005, 20: 0.01, 50: 0.025, 100 :0.02}

    backtest(
        capital=200, leverage=leverage, fee_rate=fee_rates.get(leverage),
        loss_limit=0.2, include_early=True,
        take_profit_pct=0.08, stop_loss_pct=0.02)

    # backtest(capital=100, leverage=5, fee_rate=0.0002, loss_limit=0.2,
    #          include_early=False,
    #          take_profit_pct=0.08,
    #          stop_loss_pct=0.04)
