#!/usr/bin/env python3.12
"""
Запускний скрипт (крок 7.9.4): збирає крос-секційний long-формат з уже
готових *_1d_exploration.csv per-symbol файлів (вихід split_data.py) для
всіх CROSS_SECTIONAL_SYMBOLS.

Індикатори не переносяться — крос-секційний замір (momentum за close,
ранжування між монетами) використовує лише сирі OHLCV.

Виходи:
    data/processed/crosssec_1d_exploration.csv — основний замір,
        2019-01-01 -> 2023-12-31 (юніверс стабільний: повні 22 монети
        вже з 2019 року, див. лог verify_data.py).
    data/processed/crosssec_1d_2018.csv — довідковий 2018 рік (юніверс іще
        формується: 5 -> 15 монет за рік), у основний замір НЕ йде.

Запуск (з кореня проекту crypto_bot/, після split_data.py):
    python3.12 scripts/build_cross_sectional.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.data_pipeline.cross_sectional import (  # noqa: E402
    build_cross_sectional_long,
    log_cross_sectional_summary,
)
from bot.utils.logger import get_logger  # noqa: E402
from config.settings import (  # noqa: E402
    CROSS_SECTIONAL_SYMBOLS,
    CROSS_SECTIONAL_TIMEFRAME,
    LOGS_DIR,
    PROCESSED_DATA_DIR,
)

logger = get_logger(__name__, log_dir=LOGS_DIR)

# Основний період заміру: юніверс стабільний (усі 22 монети присутні щороку).
MAIN_START_DATE = "2019-01-01"
MAIN_END_DATE = "2023-12-31"

# Довідковий 2018 рік — юніверс іще формується, у основний замір НЕ йде,
# лише для порівняльної перевірки.
REFERENCE_2018_START_DATE = "2018-01-01"
REFERENCE_2018_END_DATE = "2018-12-31"

MIN_COINS_PER_DAY = 15


def main() -> None:
    logger.info("=== Основний крос-секційний датасет: %s -> %s ===", MAIN_START_DATE, MAIN_END_DATE)
    main_df = build_cross_sectional_long(
        symbols=CROSS_SECTIONAL_SYMBOLS,
        timeframe=CROSS_SECTIONAL_TIMEFRAME,
        start_date=MAIN_START_DATE,
        end_date=MAIN_END_DATE,
        processed_data_dir=PROCESSED_DATA_DIR,
    )
    main_path = PROCESSED_DATA_DIR / "crosssec_1d_exploration.csv"
    main_df.to_csv(main_path, index=False)
    logger.info("Збережено: %s (%d рядків)", main_path, len(main_df))
    log_cross_sectional_summary(main_df, "crosssec_1d_exploration", min_coins_per_day=MIN_COINS_PER_DAY)

    logger.info(
        "=== Довідковий 2018 рік (у основний замір НЕ йде): %s -> %s ===",
        REFERENCE_2018_START_DATE, REFERENCE_2018_END_DATE,
    )
    ref_2018_df = build_cross_sectional_long(
        symbols=CROSS_SECTIONAL_SYMBOLS,
        timeframe=CROSS_SECTIONAL_TIMEFRAME,
        start_date=REFERENCE_2018_START_DATE,
        end_date=REFERENCE_2018_END_DATE,
        processed_data_dir=PROCESSED_DATA_DIR,
    )
    ref_2018_path = PROCESSED_DATA_DIR / "crosssec_1d_2018.csv"
    ref_2018_df.to_csv(ref_2018_path, index=False)
    logger.info("Збережено: %s (%d рядків)", ref_2018_path, len(ref_2018_df))
    log_cross_sectional_summary(ref_2018_df, "crosssec_1d_2018 (довідковий)", min_coins_per_day=MIN_COINS_PER_DAY)


if __name__ == "__main__":
    main()
