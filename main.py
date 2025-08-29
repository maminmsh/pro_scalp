import asyncio

from backtest import backtest_run
from main_with_sqlite import main_with_sqlite, send_telegram
from test import test_barayand_signal_change



if __name__ == "__main__":
    # not live
    print('--------start-------')
    # main_with_sqlite()
    test_barayand_signal_change()
    backtest_run()
    print('------finish-------')

    # live
    send_telegram('BBBBBBBBBBBBBBBBB')
    # asyncio.run(main_live_websocket('BTC/USDT'))

