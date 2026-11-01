#!/usr/bin/env python3.12
"""
Запускний скрипт: прогін двигуна бектесту (src/bot/backtesting/engine.py)
на baseline-стратегії EMACrossStrategy — перший тестовий стенд для
перевірки коректності самого двигуна (docs/trading_decisions.md, розділ
"Baseline"), НЕ фінальна стратегія.

КРИТИЧНО: прогін виконується ЛИШЕ на exploration-наборі
(data/processed/btc_4h_exploration.csv). Holdout-файл цей скрипт не читає
і не імпортує НІКОЛИ — див. docs/trading_decisions.md, розділ "Методологія:
розділення даних".

Запуск (з кореня проекту crypto_bot/):
    python3.12 scripts/backtest.py
    python3.12 scripts/backtest.py --live-min-notional   # опційно: реальний
                                                          # min_notional з Binance
                                                          # замість DEFAULT_MIN_NOTIONAL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# config/ не входить у встановлюваний пакет bot (лежить поруч зі src/, а не
# в ньому) — явно додаємо корінь проекту в sys.path, щоб import config.settings
# працював незалежно від того, звідки фактично запущено скрипт.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import ccxt  # noqa: E402  (імпорт після маніпуляції з sys.path — навмисно)
import pandas as pd  # noqa: E402

from bot.backtesting.engine import BacktestEngine, compute_buy_and_hold_benchmark  # noqa: E402
from bot.data_pipeline.downloader import create_exchange  # noqa: E402
from bot.risk.risk_manager import extract_min_notional_from_market  # noqa: E402
from bot.strategies.ema_cross import EMACrossStrategy  # noqa: E402
from bot.utils.logger import get_logger  # noqa: E402
from config import settings  # noqa: E402

logger = get_logger(__name__, log_dir=settings.LOGS_DIR)

EXPLORATION_FILE = settings.PROCESSED_DATA_DIR / "btc_4h_exploration.csv"
HOLDOUT_FILENAME_MARKER = "holdout"  # захист від випадкового читання holdout-файлу


def _load_exploration_data() -> pd.DataFrame:
    """
    Завантажує ЛИШЕ exploration-набір і явно захищається від помилкового
    використання holdout-файлу (docs/trading_decisions.md: holdout —
    недоторканий до фінального одноразового прогону).

    :raises FileNotFoundError: якщо data/processed/btc_4h_exploration.csv відсутній
        (спершу запусти scripts/add_indicators.py і scripts/split_data.py).
    :raises AssertionError: якщо шлях до файлу підозріло вказує на holdout —
        запобіжник від випадкової зміни EXPLORATION_FILE у майбутньому.
    """
    assert HOLDOUT_FILENAME_MARKER not in EXPLORATION_FILE.name.lower(), (
        "Захисна перевірка: шлях до даних бектесту виглядає як holdout-файл. "
        "Прогін на holdout робиться ОДИН РАЗ окремо, коли стратегія повністю "
        "зафіксована — не з цього скрипта в поточному вигляді."
    )
    if not EXPLORATION_FILE.exists():
        raise FileNotFoundError(
            f"Не знайдено {EXPLORATION_FILE}. Спершу запусти "
            f"scripts/add_indicators.py і scripts/split_data.py."
        )

    df = pd.read_csv(EXPLORATION_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(
        "Завантажено exploration-набір: %s, %d рядків, період %s -> %s",
        EXPLORATION_FILE, len(df), df["datetime"].iloc[0], df["datetime"].iloc[-1],
    )
    return df


def _resolve_min_notional(use_live: bool) -> float:
    """
    Визначає min_notional для сайзингу. За замовчуванням — offline
    (config.settings.DEFAULT_MIN_NOTIONAL), щоб скрипт працював без мережі.

    З прапорцем --live-min-notional намагається дістати реальний ліміт з
    Binance через ccxt (execution/backtesting-шар — risk_manager.py сам
    мережу не викликає, див. його докстрінг); при мережевій помилці — м'який
    fallback на DEFAULT_MIN_NOTIONAL з попередженням у логах, скрипт не падає.
    """
    if not use_live:
        logger.info(
            "min_notional: offline fallback = %.2f USDT (config.settings.DEFAULT_MIN_NOTIONAL). "
            "Для реального ліміту біржі запусти з --live-min-notional.",
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


def _print_report(
    strategy_name: str,
    metrics: dict[str, float],
    benchmark: dict[str, float],
    skipped_min_notional_count: int,
) -> str:
    """Формує текстовий звіт бектесту (лог + повернене значення для збереження у файл)."""
    beats_benchmark = metrics["total_return"] > benchmark["total_return"]
    gate_trades = metrics["trades_count"] >= 30
    gate_drawdown = metrics["max_drawdown"] < 0.25
    gate_profit_factor = metrics["profit_factor"] > 1.3
    gate_all_passed = beats_benchmark and gate_trades and gate_drawdown and gate_profit_factor

    lines = [
        "=" * 70,
        f"ЗВІТ БЕКТЕСТУ — стратегія: {strategy_name}",
        "Набір даних: EXPLORATION ONLY (btc_4h_exploration.csv, 2021-01-01 -> 2023-12-31)",
        "=" * 70,
        "",
        "--- Метрики стратегії ---",
        f"Total return:        {metrics['total_return']:>10.2%}",
        f"Final equity:         {metrics['final_equity']:>10.2f} USDT",
        f"Max drawdown:         {metrics['max_drawdown']:>10.2%}",
        f"Profit factor:        {metrics['profit_factor']:>10.2f}",
        f"Кількість угод:       {int(metrics['trades_count']):>10d}",
        f"Win rate:             {metrics['win_rate']:>10.2%}",
        f"Середній виграш:      {metrics['avg_win']:>10.4f} USDT",
        f"Середня програш:      {metrics['avg_loss']:>10.4f} USDT",
        f"Sharpe (довідковий,   {metrics['sharpe_ratio_informational_only']:>10.2f}   "
        f"НЕ gate-критерій — важкі хвости розподілу, Sharpe оптимістичний)",
        "",
        "--- Бенчмарк: buy-and-hold BTC (той самий період) ---",
        f"Total return (B&H):   {benchmark['total_return']:>10.2%}",
        f"Final equity (B&H):   {benchmark['final_equity']:>10.2f} USDT",
        f"Max drawdown (B&H):   {benchmark['max_drawdown']:>10.2%}",
        "",
        "--- Чесність виконання ---",
        f"Пропущено угод (below_min_notional): {skipped_min_notional_count}",
        "",
        "--- Gate-критерії (docs/trading_decisions.md) ---",
        f"[{'OK' if beats_benchmark else 'FAIL'}] Перевершує buy-and-hold BTC",
        f"[{'OK' if gate_trades else 'FAIL'}] Кількість угод >= 30 (фактично: {int(metrics['trades_count'])})",
        f"[{'OK' if gate_drawdown else 'FAIL'}] Max drawdown < 25% (фактично: {metrics['max_drawdown']:.2%})",
        f"[{'OK' if gate_profit_factor else 'FAIL'}] Profit factor > 1.3 (фактично: {metrics['profit_factor']:.2f})",
        "",
        f"РЕЗУЛЬТАТ: {'ВСІ GATE-КРИТЕРІЇ ПРОЙДЕНО' if gate_all_passed else 'GATE-КРИТЕРІЇ НЕ ПРОЙДЕНО'}"
        + (
            ""
            if gate_all_passed
            else " — це нормальний результат для baseline-стенду; "
            "не підганяємо параметри, повертаємось до EDA/гіпотез, якщо мета — не лише "
            "перевірка коректності двигуна."
        ),
        "=" * 70,
    ]
    report = "\n".join(lines)
    logger.info("\n%s", report)
    return report


def main() -> None:
    """Завантажує exploration-набір, прогонить EMACrossStrategy через BacktestEngine,
    друкує звіт і зберігає журнал угод + текстовий звіт у settings.RESULTS_DIR."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-min-notional",
        action="store_true",
        help="Отримати реальний min_notional з Binance через ccxt замість offline-fallback.",
    )
    args = parser.parse_args()

    df = _load_exploration_data()
    min_notional = _resolve_min_notional(use_live=args.live_min_notional)

    strategy = EMACrossStrategy()
    engine = BacktestEngine(
        strategy=strategy,
        initial_balance=settings.INITIAL_BALANCE,
        min_notional=min_notional,
    )

    result = engine.run(df)
    benchmark = compute_buy_and_hold_benchmark(df, initial_balance=settings.INITIAL_BALANCE)

    report = _print_report(
        strategy_name=strategy.name,
        metrics=result.metrics,
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
