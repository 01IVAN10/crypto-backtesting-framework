#!/usr/bin/env python3.12
"""
Запускний скрипт: читає сирі OHLCV із data/raw/, додає EDA-індикатори
(src/bot/data_pipeline/indicators.py) і зберігає результат у data/processed/.

Запуск (з кореня проекту crypto_bot/, після scripts/download_data.py):
    python3.12 scripts/add_indicators.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# config/ не входить у встановлюваний пакет bot, тому явно додаємо корінь
# проекту в sys.path — аналогічно до scripts/download_data.py.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402

from bot.data_pipeline.indicators import add_indicators  # noqa: E402
from bot.utils.logger import get_logger  # noqa: E402
from config.settings import (  # noqa: E402
    LOGS_DIR,
    PIPELINE_TARGETS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

logger = get_logger(__name__, log_dir=LOGS_DIR)


def _raw_filepath(symbol: str, timeframe: str) -> Path:
    """Шлях до сирого CSV — та сама схема іменування, що й у downloader.py."""
    return RAW_DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}.csv"


def _processed_filepath(symbol: str, timeframe: str) -> Path:
    """Шлях до обробленого CSV з індикаторами."""
    return PROCESSED_DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}.csv"


def load_raw_ohlcv(filepath: Path) -> pd.DataFrame:
    """
    Завантажує сирий OHLCV CSV, збережений downloader.py.

    :raises FileNotFoundError: якщо файл відсутній (спочатку треба запустити download_data.py)
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Сирий файл не знайдено: {filepath}. "
            f"Спочатку запусти: python3.12 scripts/download_data.py"
        )

    df = pd.read_csv(filepath)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def process_symbol(symbol: str, timeframe: str) -> Path:
    """
    Завантажує сирі дані для одного символу/таймфрейму, додає індикатори
    та зберігає результат у data/processed/.

    :return: шлях до збереженого файлу
    """
    raw_path = _raw_filepath(symbol, timeframe)
    df_raw = load_raw_ohlcv(raw_path)

    df_processed = add_indicators(df_raw)

    processed_path = _processed_filepath(symbol, timeframe)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(processed_path, index=False)
    logger.info(
        "Збережено: %s (%d рядків, %d колонок)",
        processed_path, len(df_processed), df_processed.shape[1],
    )
    return processed_path


def main() -> None:
    """Обробляє всі пари (symbol, timeframe) з config.PIPELINE_TARGETS."""
    total = len(PIPELINE_TARGETS)
    done = 0
    failed: list[tuple[str, str, str]] = []

    for symbol, timeframe in PIPELINE_TARGETS:
        done += 1
        logger.info("[%d/%d] Обробка %s / %s", done, total, symbol, timeframe)
        try:
            process_symbol(symbol, timeframe)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Не вдалося обробити %s / %s: %s", symbol, timeframe, exc)
            failed.append((symbol, timeframe, str(exc)))

    if failed:
        logger.warning("Завершено з помилками (%d/%d невдалих):", len(failed), total)
        for symbol, timeframe, reason in failed:
            logger.warning("  %s / %s -> %s", symbol, timeframe, reason)
        sys.exit(1)

    logger.info("Усі файли успішно оброблено (%d/%d).", total, total)


if __name__ == "__main__":
    main()