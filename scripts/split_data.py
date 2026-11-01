#!/usr/bin/env python3.12
"""
Запускний скрипт: читає processed-файл (з усіма індикаторами, з add_indicators.py)
і ділить його на exploration/holdout за датою (EXPLORATION_END_DATE /
HOLDOUT_START_DATE з config/settings.py), без shuffle.

Завершує дата-пайплайн перед EDA. Holdout не призначений для перегляду під
час дослідження ринку — лише для фінальної одноразової перевірки.

Запуск (з кореня проекту crypto_bot/, після scripts/add_indicators.py):
    python3.12 scripts/split_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# config/ не входить у встановлюваний пакет bot, тому явно додаємо корінь
# проекту в sys.path — аналогічно до download_data.py / add_indicators.py.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402

from bot.data_pipeline.splitter import log_split_summary, split_exploration_holdout  # noqa: E402
from bot.utils.logger import get_logger  # noqa: E402
from config.settings import (  # noqa: E402
    EXPLORATION_END_DATE,
    HOLDOUT_START_DATE,
    LOGS_DIR,
    PIPELINE_TARGETS,
    PROCESSED_DATA_DIR,
)

logger = get_logger(__name__, log_dir=LOGS_DIR)


def _processed_input_filepath(symbol: str, timeframe: str) -> Path:
    """Шлях до processed-файлу з індикаторами — те, що зберігає add_indicators.py."""
    return PROCESSED_DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}.csv"


def _split_output_filepath(symbol: str, timeframe: str, split_name: str) -> Path:
    """
    Шлях до файлу одного зі спліт-наборів.

    Коротше ім'я, ніж у processed-файлу: лише базова валюта в нижньому регістрі,
    напр. "BTC/USDT" + "4h" + "exploration" -> "btc_4h_exploration.csv".
    """
    base_currency = symbol.split("/")[0].lower()
    return PROCESSED_DATA_DIR / f"{base_currency}_{timeframe}_{split_name}.csv"


def load_processed_csv(filepath: Path) -> pd.DataFrame:
    """
    Завантажує processed CSV (з індикаторами), збережений add_indicators.py.

    :raises FileNotFoundError: якщо файл відсутній (спочатку треба запустити add_indicators.py)
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed-файл не знайдено: {filepath}. "
            f"Спочатку запусти: python3.12 scripts/add_indicators.py"
        )

    df = pd.read_csv(filepath)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def split_symbol(symbol: str, timeframe: str) -> tuple[Path, Path]:
    """
    Завантажує processed-дані для одного символу/таймфрейму, ділить на
    exploration/holdout і зберігає обидва файли.

    :return: (шлях до exploration-файлу, шлях до holdout-файлу)
    """
    input_path = _processed_input_filepath(symbol, timeframe)
    df = load_processed_csv(input_path)

    exploration_df, holdout_df = split_exploration_holdout(
        df, EXPLORATION_END_DATE, HOLDOUT_START_DATE
    )

    exploration_path = _split_output_filepath(symbol, timeframe, "exploration")
    holdout_path = _split_output_filepath(symbol, timeframe, "holdout")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    exploration_df.to_csv(exploration_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)

    logger.info("Збережено: %s (%d рядків)", exploration_path, len(exploration_df))
    logger.info("Збережено: %s (%d рядків)", holdout_path, len(holdout_df))

    # Самоперевірка для користувача: rows, перша/остання дата, NaN по колонках.
    # Вміст holdout навмисно не показується в чаті — лише ці метадані в логах.
    log_split_summary(f"{symbol} {timeframe} exploration", exploration_df)
    log_split_summary(f"{symbol} {timeframe} holdout", holdout_df)

    return exploration_path, holdout_path


def main() -> None:
    """Обробляє всі пари (symbol, timeframe) з config.PIPELINE_TARGETS."""
    total = len(PIPELINE_TARGETS)
    done = 0
    failed: list[tuple[str, str, str]] = []

    for symbol, timeframe in PIPELINE_TARGETS:
        done += 1
        logger.info("[%d/%d] Розбиття %s / %s", done, total, symbol, timeframe)
        try:
            split_symbol(symbol, timeframe)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Не вдалося розбити %s / %s: %s", symbol, timeframe, exc)
            failed.append((symbol, timeframe, str(exc)))

    if failed:
        logger.warning("Завершено з помилками (%d/%d невдалих):", len(failed), total)
        for symbol, timeframe, reason in failed:
            logger.warning("  %s / %s -> %s", symbol, timeframe, reason)
        sys.exit(1)

    logger.info("Усі файли успішно розбито (%d/%d).", total, total)


if __name__ == "__main__":
    main()