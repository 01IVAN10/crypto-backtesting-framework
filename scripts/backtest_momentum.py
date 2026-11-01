#!/usr/bin/env python3.12
"""
Запускний скрипт: ОДИН прогін Momentum30Strategy (src/bot/strategies/momentum.py)
через наявний BacktestEngine (src/bot/backtesting/engine.py) на денному
exploration-наборі. Це перша НЕ-тестова стратегія проекту — на відміну від
scripts/backtest.py (EMACrossStrategy, лише стенд для перевірки двигуна).

Виходи НЕ змінюються відносно того, що вже реалізовано в engine.py:
початковий стоп 2×ATR(14)-на-вході, трейлінг-стоп 6×max(поточний ATR,
ATR-на-вході) без порогу активації, take-profit відсутній, зворотний
сигнал стратегії. Sizing — через наявний risk/risk_manager.calculate_position_size
(викликається зсередини engine.run(), тут не дублюється).

Параметри НЕ підбираються — це один зафіксований наперед прогін
(config.settings.MOMENTUM_LOOKBACK/MOMENTUM_TERCILE_WINDOW/TERCILE_PCT).

Єдина зміна відносно engine.py, використаного для EMACrossStrategy: денним
барам відповідає інший коефіцієнт річної анулізації Sharpe (365, а не
6×365 для 4h) — передається явно через новий опційний параметр
BacktestEngine(bars_per_year=...), який ні на що інше не впливає
(див. docstring engine.py). Це ЛИШЕ довідкова метрика, не gate-критерій.

КРИТИЧНО: прогін виконується ЛИШЕ на exploration-наборі
(data/processed/btc_1d_exploration.csv). Holdout-файл цей скрипт не читає
і не імпортує НІКОЛИ (docs/trading_decisions.md, "Методологія: розділення
даних").

Запуск (з кореня проекту crypto_bot/):
    python3.12 scripts/backtest_momentum.py
    python3.12 scripts/backtest_momentum.py --live-min-notional
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import ccxt  # noqa: E402
import pandas as pd  # noqa: E402

from bot.backtesting.engine import BacktestEngine, Trade, compute_buy_and_hold_benchmark  # noqa: E402
from bot.data_pipeline.downloader import create_exchange  # noqa: E402
from bot.risk.risk_manager import extract_min_notional_from_market  # noqa: E402
from bot.strategies.momentum import Momentum30Strategy  # noqa: E402
from bot.utils.logger import get_logger  # noqa: E402
from config import settings  # noqa: E402

logger = get_logger(__name__, log_dir=settings.LOGS_DIR)

EXPLORATION_FILE = settings.PROCESSED_DATA_DIR / "btc_1d_exploration.csv"
HOLDOUT_FILENAME_MARKER = "holdout"
DAILY_BARS_PER_YEAR = 365
YEARLY_BREAKDOWN_RANGE = (2019, 2023)  # роки, обов'язкові в звіті за завданням


def _load_exploration_data() -> pd.DataFrame:
    """
    Завантажує ЛИШЕ денний exploration-набір, з тим самим захисним
    патерном, що й scripts/backtest.py (assert проти випадкового читання
    holdout-файлу).

    :raises FileNotFoundError: якщо data/processed/btc_1d_exploration.csv відсутній.
    :raises AssertionError: якщо шлях виглядає як holdout-файл.
    """
    assert HOLDOUT_FILENAME_MARKER not in EXPLORATION_FILE.name.lower(), (
        "Захисна перевірка: шлях до даних бектесту виглядає як holdout-файл. "
        "Прогін на holdout робиться ОДИН РАЗ окремо, коли стратегія повністю "
        "зафіксована — не з цього скрипта в поточному вигляді."
    )
    if not EXPLORATION_FILE.exists():
        raise FileNotFoundError(
            f"Не знайдено {EXPLORATION_FILE}. Спершу запусти "
            f"scripts/add_indicators.py і scripts/split_data.py для 1d-таймфрейму."
        )

    df = pd.read_csv(EXPLORATION_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(
        "Завантажено денний exploration-набір: %s, %d рядків, період %s -> %s",
        EXPLORATION_FILE, len(df), df["datetime"].iloc[0], df["datetime"].iloc[-1],
    )
    return df


def _resolve_min_notional(use_live: bool) -> float:
    """Те саме, що й у scripts/backtest.py — offline fallback за замовчуванням."""
    if not use_live:
        logger.info(
            "min_notional: offline fallback = %.2f USDT (config.settings.DEFAULT_MIN_NOTIONAL).",
            settings.DEFAULT_MIN_NOTIONAL,
        )
        return settings.DEFAULT_MIN_NOTIONAL

    try:
        exchange = create_exchange(settings.EXCHANGE_ID)
        markets = exchange.load_markets()
        symbol = settings.SYMBOLS[0]
        min_notional = extract_min_notional_from_market(
            markets[symbol], fallback=settings.DEFAULT_MIN_NOTIONAL
        )
        logger.info("min_notional отримано з Binance для %s: %.2f USDT", symbol, min_notional)
        return min_notional
    except (ccxt.NetworkError, ccxt.ExchangeError, KeyError) as exc:
        logger.warning(
            "Не вдалося отримати min_notional з Binance (%s) — fallback на "
            "DEFAULT_MIN_NOTIONAL=%.2f USDT.",
            exc, settings.DEFAULT_MIN_NOTIONAL,
        )
        return settings.DEFAULT_MIN_NOTIONAL


def _yearly_pnl(trades: list[Trade]) -> pd.Series:
    """
    Порічна розбивка net_pnl за роком ВИХОДУ з угоди (коли P&L фактично
    реалізовано), не роком входу. Порожній список угод -> порожній Series.
    """
    if not trades:
        return pd.Series(dtype=float)
    exit_years = pd.Series([pd.Timestamp(t.exit_time).year for t in trades])
    net_pnls = pd.Series([t.net_pnl for t in trades])
    return net_pnls.groupby(exit_years).sum().sort_index()


def _time_in_market_pct(trades: list[Trade], total_bars: int) -> float:
    """
    Частка барів, коли позиція була відкрита: сума bars_held (з
    інструментації Trade, engine.py) поділена на загальну кількість барів
    у прогоні. Наближення (bars_held рахує від entry_idx до exit_idx
    включно, тобто позиція технічно "відкрита" в цьому діапазоні) —
    достатнє для порівняння з очікуваними ~36%.
    """
    if total_bars <= 0:
        return 0.0
    bars_in_position = sum(t.bars_held for t in trades)
    return bars_in_position / total_bars


def _cagr(initial_equity: float, final_equity: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Annualized-дохідність (CAGR) за фактичну довжину періоду прогону (365.25 днів/рік)."""
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    years = days / 365.25
    if years <= 0 or initial_equity <= 0:
        return 0.0
    return (final_equity / initial_equity) ** (1.0 / years) - 1.0


def _calmar(cagr: float, max_drawdown: float) -> float:
    """Calmar = CAGR / max drawdown. Довідковий показник (не gate-критерій)."""
    if max_drawdown <= 0:
        return float("inf")
    return cagr / max_drawdown


def _print_report(
    strategy_name: str,
    df: pd.DataFrame,
    result_metrics: dict[str, float],
    trades: list[Trade],
    equity_curve: pd.Series,
    benchmark: dict[str, float],
    skipped_min_notional_count: int,
) -> str:
    """Формує розширений текстовий звіт бектесту momentum-стратегії."""
    beats_benchmark = result_metrics["total_return"] > benchmark["total_return"]
    gate_trades = result_metrics["trades_count"] >= 30
    gate_drawdown = result_metrics["max_drawdown"] < 0.25
    gate_profit_factor = result_metrics["profit_factor"] > 1.3
    gate_all_passed = beats_benchmark and gate_trades and gate_drawdown and gate_profit_factor

    start_time = df["datetime"].iloc[0]
    end_time = df["datetime"].iloc[-1]

    strategy_cagr = _cagr(settings.INITIAL_BALANCE, result_metrics["final_equity"], start_time, end_time)
    benchmark_cagr = _cagr(settings.INITIAL_BALANCE, benchmark["final_equity"], start_time, end_time)
    strategy_calmar = _calmar(strategy_cagr, result_metrics["max_drawdown"])
    benchmark_calmar = _calmar(benchmark_cagr, benchmark["max_drawdown"])

    yearly = _yearly_pnl(trades)
    time_in_market = _time_in_market_pct(trades, total_bars=len(df))

    yearly_lines = ["--- P&L by year (net, USDT, by trade EXIT year) ---"]
    if yearly.empty:
        yearly_lines.append("  (no trades)")
    else:
        for year, pnl in yearly.items():
            marker = " <-- pre-registered expectation: a visible loss in 2022 (bear year)" if year == 2022 else ""
            in_range = "" if YEARLY_BREAKDOWN_RANGE[0] <= year <= YEARLY_BREAKDOWN_RANGE[1] else "  (outside 2019-2023)"
            yearly_lines.append(f"  {year}: {pnl:>10.4f} USDT{in_range}{marker}")
        missing_years = [
            y for y in range(YEARLY_BREAKDOWN_RANGE[0], YEARLY_BREAKDOWN_RANGE[1] + 1)
            if y not in yearly.index
        ]
        if missing_years:
            yearly_lines.append(f"  (no trades in: {missing_years} — zero trades that year, not an error)")

    lines = [
        "=" * 70,
        f"BACKTEST REPORT — strategy: {strategy_name} (first non-test strategy)",
        "Dataset: EXPLORATION ONLY (btc_1d_exploration.csv, daily timeframe)",
        f"Period: {start_time} -> {end_time} ({len(df)} daily bars)",
        "Parameters were NOT tuned: a single run, fixed in advance.",
        "=" * 70,
        "",
        "--- Strategy metrics ---",
        f"Total return:          {result_metrics['total_return']:>10.2%}",
        f"Final equity:           {result_metrics['final_equity']:>10.2f} USDT",
        f"Max drawdown:           {result_metrics['max_drawdown']:>10.2%}",
        f"Profit factor:          {result_metrics['profit_factor']:>10.2f}",
        f"Trades:                 {int(result_metrics['trades_count']):>10d}",
        f"Skipped (below_min_notional):   {skipped_min_notional_count:>3d}",
        f"Win rate:               {result_metrics['win_rate']:>10.2%}",
        f"Average win:            {result_metrics['avg_win']:>10.4f} USDT",
        f"Average loss:           {result_metrics['avg_loss']:>10.4f} USDT",
        f"Sharpe (reference)      {result_metrics['sharpe_ratio_informational_only']:>10.2f}   "
        f"NOT a gate criterion; annualised at 365 bars/year for 1d",
        f"Time in market:         {time_in_market:>10.2%}  (fitness study estimate ~36%;"
        f" may differ with the rolling threshold)",
        "",
        "--- Calmar (CAGR / max drawdown) — reference, not a gate ---",
        f"Strategy:     CAGR={strategy_cagr:>8.2%}  MaxDD={result_metrics['max_drawdown']:>7.2%}  Calmar={strategy_calmar:>7.2f}",
        f"Buy-and-hold: CAGR={benchmark_cagr:>8.2%}  MaxDD={benchmark['max_drawdown']:>7.2%}  Calmar={benchmark_calmar:>7.2f}",
        "",
        *yearly_lines,
        "",
        "--- Benchmark: buy-and-hold BTC (same period) ---",
        f"Total return (B&H):     {benchmark['total_return']:>10.2%}",
        f"Final equity (B&H):     {benchmark['final_equity']:>10.2f} USDT",
        f"Max drawdown (B&H):     {benchmark['max_drawdown']:>10.2%}",
        "",
        "--- Gate criteria (docs/trading_decisions.md) ---",
        f"[{'OK' if beats_benchmark else 'FAIL'}] Beats buy-and-hold BTC",
        f"[{'OK' if gate_trades else 'FAIL'}] Number of trades >= 30 (actual: {int(result_metrics['trades_count'])})",
        f"[{'OK' if gate_drawdown else 'FAIL'}] Max drawdown < 25% (actual: {result_metrics['max_drawdown']:.2%})",
        f"[{'OK' if gate_profit_factor else 'FAIL'}] Profit factor > 1.3 (actual: {result_metrics['profit_factor']:.2f})",
        "",
        f"RESULT: {'ALL GATE CRITERIA PASSED' if gate_all_passed else 'GATE CRITERIA NOT PASSED'}"
        + (
            ""
            if gate_all_passed
            else " — parameters are NOT tuned; a failed strategy means a new hypothesis "
            "as a separate step, not changing these numbers after the fact."
        ),
        "=" * 70,
    ]
    report = "\n".join(lines)
    logger.info("\n%s", report)
    return report


def main() -> None:
    """Один прогін Momentum30Strategy на денному exploration-наборі, звіт + журнал угод у results/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-min-notional",
        action="store_true",
        help="Отримати реальний min_notional з Binance через ccxt замість offline-fallback.",
    )
    args = parser.parse_args()

    df = _load_exploration_data()
    min_notional = _resolve_min_notional(use_live=args.live_min_notional)

    strategy = Momentum30Strategy()
    engine = BacktestEngine(
        strategy=strategy,
        initial_balance=settings.INITIAL_BALANCE,
        min_notional=min_notional,
        bars_per_year=DAILY_BARS_PER_YEAR,  # ЛИШЕ анулізація Sharpe; жодна торгова логіка не змінена
    )

    result = engine.run(df)
    benchmark = compute_buy_and_hold_benchmark(df, initial_balance=settings.INITIAL_BALANCE)

    report = _print_report(
        strategy_name=strategy.name,
        df=df,
        result_metrics=result.metrics,
        trades=result.trades,
        equity_curve=result.equity_curve,
        benchmark=benchmark,
        skipped_min_notional_count=result.skipped_min_notional_count,
    )

    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trades_path = settings.RESULTS_DIR / f"backtest_{strategy.name}_trades.csv"
    report_path = settings.RESULTS_DIR / f"backtest_{strategy.name}_report.txt"

    result.save_trades_csv(str(trades_path))
    report_path.write_text(report, encoding="utf-8")
    logger.info("Текстовий звіт збережено: %s", report_path)


if __name__ == "__main__":
    main()
