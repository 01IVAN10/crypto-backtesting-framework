#!/usr/bin/env python3.12
"""
Запускний скрипт: завантажує історичні OHLCV-дані для всіх символів і таймфреймів
з config/settings.py та зберігає їх у data/raw/.

Запуск (з кореня проекту crypto_bot/):
    python3.12 scripts/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# config/ не входить у встановлюваний пакет bot (він лежить поруч зі src/, а не в ньому),
# тому явно додаємо корінь проекту в sys.path — це дозволяє імпортувати config.settings
# незалежно від того, з якого каталогу фактично запущено скрипт.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import ccxt  # noqa: E402  (імпорт після маніпуляції з sys.path — навмисно)

from bot.data_pipeline.downloader import create_exchange, download_symbol_data  # noqa: E402
from bot.utils.logger import get_logger  # noqa: E402
from config.settings import (  # noqa: E402
    CROSS_SECTIONAL_SYMBOLS,
    DOWNLOAD_SINCE_DATE,
    EXCHANGE_ID,
    LOGS_DIR,
    PIPELINE_TARGETS,
    RAW_DATA_DIR,
)

logger = get_logger(__name__, log_dir=LOGS_DIR)


def main() -> None:
    """Завантажує дані для всіх пар (symbol, timeframe) з config.PIPELINE_TARGETS."""
    logger.warning(
        "SURVIVORSHIP BIAS: CROSS_SECTIONAL_SYMBOLS (%s) складено з монет, що "
        "торгуються СЬОГОДНІ. Монети, що зникли/втратили лістинг Binance за "
        "2018-2023, у наборі відсутні — це систематично завищує результат "
        "крос-секційної стратегії. Повністю прибрати без історичних списків "
        "лістингу неможливо; враховуй це при інтерпретації результатів кроку 7.9.4.",
        ", ".join(CROSS_SECTIONAL_SYMBOLS),
    )

    exchange = create_exchange(EXCHANGE_ID)

    total = len(PIPELINE_TARGETS)
    done = 0
    failed: list[tuple[str, str, str]] = []

    for symbol, timeframe in PIPELINE_TARGETS:
        done += 1
        logger.info("[%d/%d] Завантаження %s / %s", done, total, symbol, timeframe)
        try:
            download_symbol_data(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                since_date=DOWNLOAD_SINCE_DATE,
                raw_data_dir=RAW_DATA_DIR,
            )
        except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, RuntimeError) as exc:
            logger.error("Не вдалося завантажити %s / %s: %s", symbol, timeframe, exc)
            failed.append((symbol, timeframe, str(exc)))

    if failed:
        logger.warning("Завершено з помилками (%d/%d невдалих):", len(failed), total)
        for symbol, timeframe, reason in failed:
            logger.warning("  %s / %s -> %s", symbol, timeframe, reason)
        sys.exit(1)

    logger.info("Усі дані успішно завантажено (%d/%d).", total, total)


if __name__ == "__main__":
    main()