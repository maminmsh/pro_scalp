import asyncio

import pandas as pd

from backtest import backtest_run
from config_find import run_optimizer, apply_best_and_trade
from main_with_sqlite import main_with_sqlite, send_telegram
from test import test_barayand_signal_change
import get_data as gd



if __name__ == "__main__":
    # not live
    # print('--------start-------')
    # main_with_sqlite()
    # test_barayand_signal_change()
    # backtest_run()
    # print('------finish-------')




    df_15m =gd.get_historical_ohlcv_df(timeframe='15m')
    df_1h =gd.get_historical_ohlcv_df(timeframe='1h')

    best = run_optimizer(df_1h, df_ltf=df_15m, base_tf_minutes=60, trials=80, use_bayes=False)
    print(best['best_params'])
    print(best['report'])
    signals_live = apply_best_and_trade(df_1h, best['best_params'], df_ltf=df_15m)
    print(signals_live)


    # live
    send_telegram('XXXXXXXXXXXXXXXXXXXXXXX')
    # asyncio.run(main_live_websocket('BTC/USDT'))





