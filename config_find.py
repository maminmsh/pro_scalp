"""
AI-driven config tuner and multi-timeframe decision layer
for your Pro Scalper AI (pandas + ta)

Assumptions:
- You already have `pro_scalper_ai(df, htf_data=None)` and a `config` module (as you shared).
- This script will try parameter sets, backtest them with a simple ATR-based risk model,
  do walk-forward evaluation, and pick the best config.
- It also supports a multi-timeframe ensemble decision layer (LTF + BTF + HTF agreement/vote).

How to use (quick):
1) Provide your base timeframe DataFrame `df_base` (e.g., 1H OHLCV with columns: open, high, low, close, volume).
2) Optionally provide a lower TF (e.g., 15m) DataFrame `df_ltf` aligned to the same symbol.
3) Run `best = run_optimizer(df_base, df_ltf=df_ltf, base_tf_minutes=60)`
4) `best['best_params']` will contain the suggested config, and `best['report']` has metrics.
5) To run live, call `apply_best_and_trade(df_base, best['best_params'], df_ltf=df_ltf)`.

Notes:
- If `optuna` is installed, we use Bayesian optimization. If not, we fall back to random search.
- The backtest is intentionally simple & transparent (1 position at a time, TP/SL via ATR, optional time exit).
- Replace or extend the risk model to match your production logic.
"""

from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    import optuna
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

from ta.volatility import AverageTrueRange

# ====== IMPORT YOUR MODULES ======
# They must be importable in your environment.
import config  # your config file
from pro_scalp import pro_scalper_ai  # <-- change to actual import path


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV to a new timeframe.
    rule examples: '15T', '1H', '4H', '1D'"""
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    out = df.resample(rule).agg(agg).dropna()
    return out


def set_config(params: Dict[str, Any]) -> None:
    """Apply a params dict to your `config` module dynamically."""
    for k, v in params.items():
        if hasattr(config, k):
            setattr(config, k, v)


def default_param_space(base_tf_minutes: int = 60) -> Dict[str, Any]:
    """Define a reasonable search space for 1H by default.
    You can tweak ranges as you like.
    """
    return {
        # Higher Timeframe Filter
        'useHTF': [True, False],
        'htf_tf': ['120', '240', '360'],  # 2H, 4H, 6H

        # Indicator Lengths
        'adxLen': list(range(8, 25)),
        'diLen': list(range(8, 25)),
        'atrLen': list(range(8, 25)),
        'hmaLen': [21, 34, 55],

        # Momentum Mode
        'momChoice': ["Stochastic RSI", "Fisher RSI", "Williams %R"],

        # Stoch RSI
        'stochRsiLen': list(range(9, 21)),
        'stochKLen': [2, 3, 4],
        'stochDLen': [2, 3, 4],

        # Fisher/Williams
        'fisherRsiLen': list(range(10, 21)),
        'wprLen': list(range(9, 21)),

        # TDI
        'tdiRsiLen': list(range(10, 21)),
        'tdiSignalLen': list(range(5, 11)),

        # Weights
        'wTrend': [0.3, 0.4, 0.5, 0.6],
        'wMomentum': [0.1, 0.2, 0.3, 0.4],
        'wVolatility': [0.1, 0.2, 0.3],
        'wVolume': [0.0, 0.1, 0.2, 0.3],

        # Thresholds
        'baseUpper': [20.0, 25.0, 30.0],
        'baseLower': [-20.0, -25.0, -30.0],
        'dynMultiplier': [0.3, 0.5, 0.7],
        'oscLookback': [30, 50, 80],

        # Hysteresis
        'useHysteresis': [True, False],
        'baseHystGap': [0.5, 1.0, 1.5],
        'hystMultiplier': [0.5, 1.0, 1.5],

        # Dynamic Weighting
        'useDynamicW': [True, False],
        'maxVolFactor': [1.5, 2.0, 2.5],
        'maxTrendFactor': [1.2, 1.5, 2.0],

        # Volume
        'volLookback': [30, 50, 80],

        # AI Forecast
        'useAI': [False, True],
        'ai_length': [14, 20, 30],
        'ai_future': [1, 2, 3],
        'aiAdjustment': [1.0, 1.5, 2.0],
        'aiSmoothing': [True, False],

        # Latching
        'useLatching': [True],

        # Decision layer (new)
        'trade_only_strong': [True, False],
        'risk_rr': [1.5, 2.0, 2.5],  # TP = rr * risk
        'risk_atr_mult': [1.0, 1.5, 2.0],
        'time_exit_bars': [None, 48, 96],  # optional time-based exit

        # Multi-timeframe ensemble
        'use_ltf': [False, True],
        'ltf_minutes': [15, 30],
        'ltf_weight': [0.0, 0.25, 0.5, 1.0],  # vote weight
        'ensemble_mode': ['agree', 'vote'],  # require agreement vs weighted vote
    }


# ─────────────────────────────────────────────
# Backtest + Decision Logic
# ─────────────────────────────────────────────

@dataclass
class BTResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]


def _build_signals(df_base: pd.DataFrame,
                   params: Dict[str, Any],
                   df_ltf: Optional[pd.DataFrame] = None,
                   base_tf_minutes: int = 60) -> pd.DataFrame:
    """Run Pro Scalper with given params and optionally ensemble with a lower timeframe run."""
    # Configure HTF dataset if needed
    htf_rule = None
    if params.get('useHTF', False):
        tf_map = {
            '120': '120T', '240': '240T', '360': '360T',
        }
        htf_rule = tf_map.get(str(params.get('htf_tf', '240')), '240T')
        df_htf = resample_ohlcv(df_base, htf_rule)
    else:
        df_htf = None

    # Apply params to config
    set_config(params)

    # Base timeframe signals
    base = pro_scalper_ai(df_base, htf_data=df_htf)
    base = base.copy()
    base['base_signal'] = base['stored_signal']

    # Optional LTF ensemble
    if params.get('use_ltf', False) and df_ltf is not None:
        # run same config on ltf (no extra HTF for ltf to simplify)
        set_config(params)  # ensure same state
        ltf = pro_scalper_ai(df_ltf)
        # align onto base index by last-known value (ffill after reindex)
        ltf_sig = ltf['stored_signal'].reindex(base.index, method='pad')
        base['ltf_signal'] = ltf_sig
    else:
        base['ltf_signal'] = np.nan

    # Decision layer
    trade_only_strong = params.get('trade_only_strong', True)
    mode = params.get('ensemble_mode', 'agree')

    def is_long(sig: str) -> bool:
        if trade_only_strong:
            return sig == 'Strong Buy'
        return sig in ('Strong Buy', 'Early Buy')

    def is_short(sig: str) -> bool:
        if trade_only_strong:
            return sig == 'Strong Sell'
        return sig in ('Strong Sell', 'Early Sell')

    longs = base['base_signal'].apply(is_long)
    shorts = base['base_signal'].apply(is_short)

    if params.get('use_ltf', False) and base['ltf_signal'].notna().any():
        l_longs = base['ltf_signal'].apply(lambda s: is_long(s) if isinstance(s, str) else False)
        l_shorts = base['ltf_signal'].apply(lambda s: is_short(s) if isinstance(s, str) else False)
        if mode == 'agree':
            longs = longs & l_longs
            shorts = shorts & l_shorts
        else:  # vote
            w = params.get('ltf_weight', 0.5)
            base_vote = longs.astype(float) - shorts.astype(float)  # +1 long, -1 short, 0 none
            ltf_vote = w * (l_longs.astype(float) - l_shorts.astype(float))
            vote = base_vote + ltf_vote
            longs = vote > 0.5
            shorts = vote < -0.5

    base['go_long'] = longs
    base['go_short'] = shorts

    return base


def backtest_signals(df: pd.DataFrame,
                     params: Dict[str, Any],
                     start_cash: float = 10_000.0,
                     fee: float = 0.0005) -> BTResult:
    """Simple 1x position backtest with ATR-based SL/TP and optional time exit."""
    rr = float(params.get('risk_rr', 2.0))
    atr_mult = float(params.get('risk_atr_mult', 1.5))
    time_exit = params.get('time_exit_bars', None)

    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=int(config.atrLen)).average_true_range()
    atr = atr.reindex(df.index).ffill()

    equity = []
    cash = start_cash
    pos = 0  # +1 long, -1 short, 0 flat
    entry = sl = tp = None
    bars_in_trade = 0
    trades = []

    for ts, row in df.iterrows():
        price = float(row['close'])
        signal_long = bool(row['go_long'])
        signal_short = bool(row['go_short'])
        a = float(atr.loc[ts]) if not math.isnan(float(atr.loc[ts])) else 0.0

        if pos == 0:
            # enter if signal
            if signal_long:
                pos = +1
                entry = price
                sl = entry - atr_mult * a
                tp = entry + rr * (entry - sl)
                bars_in_trade = 0
                trades.append({'time': ts, 'type': 'BUY', 'price': entry})
                cash *= (1 - fee)
            elif signal_short:
                pos = -1
                entry = price
                sl = entry + atr_mult * a
                tp = entry - rr * (sl - entry)
                bars_in_trade = 0
                trades.append({'time': ts, 'type': 'SELL', 'price': entry})
                cash *= (1 - fee)
        else:
            bars_in_trade += 1
            # check exits
            hit_tp = False
            hit_sl = False
            if pos == +1:
                if row['high'] >= tp:
                    hit_tp = True
                elif row['low'] <= sl:
                    hit_sl = True
            elif pos == -1:
                if row['low'] <= tp:
                    hit_tp = True
                elif row['high'] >= sl:
                    hit_sl = True

            exit_reason = None
            if hit_tp:
                pnl = rr  # measured in ATR-risk units; we’ll translate to equity multiplicative
                cash *= (1 + 0.01 * pnl)  # simplistic: 1% per ATR-risk unit; replace with sizing logic
                cash *= (1 - fee)
                trades.append({'time': ts, 'type': 'EXIT_TP', 'price': price})
                pos = 0; entry = sl = tp = None
                exit_reason = 'TP'
            elif hit_sl:
                pnl = -1
                cash *= (1 + 0.01 * pnl)
                cash *= (1 - fee)
                trades.append({'time': ts, 'type': 'EXIT_SL', 'price': price})
                pos = 0; entry = sl = tp = None
                exit_reason = 'SL'
            elif time_exit is not None and bars_in_trade >= int(time_exit):
                # time-based exit at close
                pnl = (price - entry) / (atr_mult * a) if pos == +1 else (entry - price) / (atr_mult * a)
                pnl = max(min(pnl, rr), -1)  # clamp to [-1, rr]
                cash *= (1 + 0.01 * pnl)
                cash *= (1 - fee)
                trades.append({'time': ts, 'type': 'EXIT_TIME', 'price': price})
                pos = 0; entry = sl = tp = None
                exit_reason = 'TIME'

        equity.append((ts, cash))

    eq = pd.Series({ts: val for ts, val in equity})

    # Metrics
    rets = eq.pct_change().fillna(0)
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * math.sqrt(252) if rets.std() > 0 else 0.0
    cumret = eq.iloc[-1] / eq.iloc[0] - 1
    dd = (eq / eq.cummax() - 1).min()

    report = {
        'final_equity': float(eq.iloc[-1]),
        'return': float(cumret),
        'sharpe': float(sharpe),
        'max_drawdown': float(dd),
        'num_trades': int(sum(1 for t in trades if t['type'] in ('EXIT_TP','EXIT_SL','EXIT_TIME'))),
    }

    trades_df = pd.DataFrame(trades)
    return BTResult(equity=eq, trades=trades_df, metrics=report)


# ─────────────────────────────────────────────
# Walk-forward Optimization
# ─────────────────────────────────────────────

def walk_forward_eval(df_base: pd.DataFrame,
                      params: Dict[str, Any],
                      df_ltf: Optional[pd.DataFrame] = None,
                      base_tf_minutes: int = 60,
                      n_splits: int = 4,
                      test_ratio: float = 0.25) -> Dict[str, float]:
    """Time-series split: evaluate params on last test chunk of each split and average metrics."""
    n = len(df_base)
    fold_metrics = []
    for i in range(n_splits):
        cut = int(n * (1 - test_ratio * (i+1)))
        if cut < int(n * 0.3):
            break
        train = df_base.iloc[:cut]
        test = df_base.iloc[cut:]
        sig_train = _build_signals(train, params, df_ltf=df_ltf, base_tf_minutes=base_tf_minutes)
        _ = backtest_signals(sig_train, params)  # fit-free, we just warm up indicators
        sig_test = _build_signals(test, params, df_ltf=df_ltf, base_tf_minutes=base_tf_minutes)
        bt = backtest_signals(sig_test, params)
        fold_metrics.append(bt.metrics)

    if not fold_metrics:
        return {'score': -1e9, 'sharpe': -1e9, 'return': -1e9, 'max_drawdown': 0.0, 'num_trades': 0}

    # Aggregate
    avg_sharpe = float(np.mean([m['sharpe'] for m in fold_metrics]))
    avg_ret = float(np.mean([m['return'] for m in fold_metrics]))
    avg_dd = float(np.mean([m['max_drawdown'] for m in fold_metrics]))

    # Composite score (maximize): Sharpe + Return - Drawdown penalty
    score = avg_sharpe + 2.0 * avg_ret + 0.5 * avg_dd  # dd is negative; adds penalty

    return {'score': score, 'sharpe': avg_sharpe, 'return': avg_ret, 'max_drawdown': avg_dd,
            'num_trades': int(np.mean([m['num_trades'] for m in fold_metrics]))}


# ─────────────────────────────────────────────
# Optimizers
# ─────────────────────────────────────────────

def _sample_params(space: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    for k, v in space.items():
        if isinstance(v, list):
            params[k] = random.choice(v)
        elif isinstance(v, tuple) and len(v) == 2:
            lo, hi = v
            if isinstance(lo, int) and isinstance(hi, int):
                params[k] = random.randint(lo, hi)
            else:
                params[k] = random.uniform(float(lo), float(hi))
        else:
            params[k] = v
    # Ensure weights sum sanity (not required by your code, but keep reasonable)
    total_w = params['wTrend'] + params['wMomentum'] + params['wVolatility'] + params['wVolume']
    if total_w == 0:
        params['wTrend'] = 0.4; params['wMomentum'] = 0.3; params['wVolatility'] = 0.2; params['wVolume'] = 0.1
    return params


def random_search(df_base: pd.DataFrame,
                  df_ltf: Optional[pd.DataFrame],
                  base_tf_minutes: int,
                  space: Dict[str, Any],
                  trials: int = 60) -> Tuple[Dict[str, Any], Dict[str, float]]:
    best_params = None
    best_metrics = {'score': -1e18}
    for i in range(trials):
        p = _sample_params(space)
        metrics = walk_forward_eval(df_base, p, df_ltf=df_ltf, base_tf_minutes=base_tf_minutes)
        if metrics['score'] > best_metrics['score']:
            best_metrics = metrics
            best_params = p
    return best_params, best_metrics


def bayes_opt(df_base: pd.DataFrame,
              df_ltf: Optional[pd.DataFrame],
              base_tf_minutes: int,
              space: Dict[str, Any],
              trials: int = 60) -> Tuple[Dict[str, Any], Dict[str, float]]:
    assert _HAS_OPTUNA, "Optuna not installed."

    def objective(trial: optuna.Trial):
        p = {}
        for k, v in space.items():
            if isinstance(v, list):
                p[k] = trial.suggest_categorical(k, v)
            elif isinstance(v, tuple) and len(v) == 2:
                lo, hi = v
                if isinstance(lo, int) and isinstance(hi, int):
                    p[k] = trial.suggest_int(k, lo, hi)
                else:
                    p[k] = trial.suggest_float(k, float(lo), float(hi))
            else:
                p[k] = v
        m = walk_forward_eval(df_base, p, df_ltf=df_ltf, base_tf_minutes=base_tf_minutes)
        return -m['score']  # minimize

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    best_params = best.params
    best_metrics = walk_forward_eval(df_base, best_params, df_ltf=df_ltf, base_tf_minutes=base_tf_minutes)
    return best_params, best_metrics


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def run_optimizer(df_base: pd.DataFrame,
                  df_ltf: Optional[pd.DataFrame] = None,
                  base_tf_minutes: int = 60,
                  trials: int = 60,
                  use_bayes: bool = False,
                  space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if space is None:
        space = default_param_space(base_tf_minutes)

    if use_bayes and _HAS_OPTUNA:
        best_params, best_metrics = bayes_opt(df_base, df_ltf, base_tf_minutes, space, trials=trials)
    else:
        best_params, best_metrics = random_search(df_base, df_ltf, base_tf_minutes, space, trials=trials)

    # Final backtest with best params on full period
    signals = _build_signals(df_base, best_params, df_ltf=df_ltf, base_tf_minutes=base_tf_minutes)
    bt = backtest_signals(signals, best_params)

    return {
        'best_params': best_params,
        'validation': best_metrics,
        'report': bt.metrics,
        'equity': bt.equity,
        'trades': bt.trades,
    }


def apply_best_and_trade(df_base: pd.DataFrame,
                         best_params: Dict[str, Any],
                         df_ltf: Optional[pd.DataFrame] = None,
                         base_tf_minutes: int = 60) -> pd.DataFrame:
    """Return a DataFrame with actionable long/short booleans per bar using the best params."""
    sig = _build_signals(df_base, best_params, df_ltf=df_ltf, base_tf_minutes=base_tf_minutes)
    return sig[['go_long', 'go_short', 'base_signal', 'ltf_signal']]


# ─────────────────────────────────────────────
# Example (commented)
# ─────────────────────────────────────────────
# if __name__ == '__main__':
#     # Load your dataframes here (ensure DateTimeIndex in UTC or consistent tz)
#     # df_1h = ...
#     # df_15m = ...
#     # result = run_optimizer(df_1h, df_ltf=df_15m, base_tf_minutes=60, trials=80, use_bayes=False)
#     # print(result['best_params'])
#     # print(result['report'])
#     pass
