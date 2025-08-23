# محاسبه مجموع مثبت‌ها، منفی‌ها و برایند کلی
import sqlite3

import pandas as pd


def load_from_sqlite(db_name="trading_data.db", table_name="signal_changes"):
    conn = sqlite3.connect(db_name)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn, index_col='timestamp')
    conn.close()
    return df

if __name__ == "__main__":

    df = load_from_sqlite()
    price_pct_change = df['price_pct_change']

    values = price_pct_change

    positive_sum = sum(x for x in values if x > 0)
    negative_sum = sum(x for x in values if x < 0)
    net_sum = positive_sum + negative_sum

    print(f"مجموع مثبت‌ها: {positive_sum:.6f}")
    print(f"مجموع منفی‌ها: {negative_sum:.6f}")
    print(f"برایند کلی: {net_sum:.6f}")
