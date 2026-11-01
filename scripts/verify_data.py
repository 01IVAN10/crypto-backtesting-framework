#!/usr/bin/env python3.12
"""
Діагностичний скрипт (одноразова перевірка, НЕ постійна стадія пайплайна):
звіряє цілісність сирих (data/raw/) і processed (data/processed/) OHLCV-файлів
для всіх пар (symbol, timeframe) з config.PIPELINE_TARGETS.

Для кожного файлу перевіряє й логує:
    - кількість рядків, перша й остання дата
    - кількість NaN по колонках (для processed — окремо ДО і ПІСЛЯ
      warm-up-періоду; після прогріву NaN не повинно бути)
    - пропущені інтервали в сітці часу (крок > очікуваного для таймфрейму)
    - окремо: чи є розрив/аномалія кроку на межі 2020-12-31 -> 2021-01-01,
      де раніше починалась історія (DOWNLOAD_SINCE_DATE був 2021-01-01
      до розширення назад до старту торгів)
    - STALENESS: чи не обривається серія РАНІШЕ сьогодні (делістинг/зупинка
      торгів) — для 1d допуск 2 дні відставання від поточної дати
    - чи перша дата серії не пізніша за EXPLORATION_END_DATE (інакше монета
      не має жодної exploration-історії — вся серія потрапляє в holdout)
    - застереження, якщо файл містить дані 2017-2020 (структурно інший ринок)

Наприкінці — підсумкова таблиця по кожній крос-секційній монеті (перша/остання
дата, бари, NaN після прогріву) і окремо покриття юніверсу по роках
(2017-2023) для оцінки статистичної потужності крос-секційного заміру.

Holdout (2024+) тут НЕ аналізується — перевіряються лише raw і processed,
що містять повну серію без розділення на exploration/holdout.

Запуск (з кореня проекту crypto_bot/, після download_data.py і add_indicators.py):
    python3.12 scripts/verify_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402

from bot.utils.logger import get_logger  # noqa: E402
from config.settings import (  # noqa: E402
    CROSS_SECTIONAL_SYMBOLS,
    CROSS_SECTIONAL_TIMEFRAME,
    EXPLORATION_END_DATE,
    LOGS_DIR,
    PIPELINE_TARGETS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

logger = get_logger(__name__, log_dir=LOGS_DIR)

# Очікуваний крок сітки часу для кожного таймфрейму — використовується для
# пошуку пропущених інтервалів (діри між сусідніми барами) і для оцінки
# кількості пропущених барів при staleness-перевірці.
EXPECTED_STEP: dict[str, pd.Timedelta] = {
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(hours=24),
}

# Максимально прийнятне відставання останньої дати серії від "сьогодні".
# Перевищення — ознака делістингу/зупинки торгів (серія ОБРИВАЄТЬСЯ), а не
# звичайна затримка оновлення даних. Для 1d поріг заданий явно користувачем.
STALENESS_TOLERANCE: dict[str, pd.Timedelta] = {
    "4h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=2),
}

# Межа, де раніше (до розширення історії до 2017-08-17) починались дані —
# перевіряємо тут окремо на розрив/аномалію кроку.
LEGACY_HISTORY_BOUNDARY = pd.Timestamp("2021-01-01", tz="UTC")

# Період 2017-2020 структурно інший ринок (менша ліквідність, менше
# інституційних грошей) — застереження для інтерпретації, не помилка даних.
EARLY_PERIOD_CAUTION_END = pd.Timestamp("2020-12-31", tz="UTC")

# Якщо перша дата серії пізніша за цю межу — у монети немає жодної
# exploration-історії (уся серія потрапляє в holdout).
LATEST_ACCEPTABLE_FIRST_DATE = pd.Timestamp(EXPLORATION_END_DATE, tz="UTC")

# Довжина найдовшого warm-up-вікна серед індикаторів (ema_200) — рядки до
# цього індексу можуть законно містити NaN.
INDICATOR_WARMUP_BARS = 200

# Діапазон років для таблиці покриття юніверсу (оцінка статистичної потужності).
COVERAGE_YEAR_START = 2017
COVERAGE_YEAR_END = 2023


def _load_csv(filepath: Path) -> pd.DataFrame | None:
    """Завантажує CSV, повертає None (з логом помилки), якщо файл відсутній."""
    if not filepath.exists():
        logger.error("Файл не знайдено: %s", filepath)
        return None
    df = pd.read_csv(filepath)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def _check_missing_intervals(df: pd.DataFrame, timeframe: str, label: str) -> None:
    """Шукає пропущені інтервали в сітці часу (крок між сусідніми барами > очікуваного)."""
    expected_step = EXPECTED_STEP.get(timeframe)
    if expected_step is None:
        logger.warning("%s: невідомий очікуваний крок для таймфрейму '%s' — перевірку сітки пропущено", label, timeframe)
        return

    diffs = df["datetime"].diff()
    gap_mask = diffs > expected_step
    n_gaps = int(gap_mask.sum())

    if n_gaps == 0:
        logger.info("%s: пропущених інтервалів немає (крок скрізь == %s)", label, expected_step)
        return

    logger.warning(
        "%s: знайдено %d пропущених інтервалів у сітці (крок > %s). Найбільший розрив: %s",
        label, n_gaps, expected_step, diffs[gap_mask].max(),
    )
    gap_positions = df.index[gap_mask.values][:10]
    for pos in gap_positions:
        gap_start = df["datetime"].iloc[pos - 1]
        gap_end = df["datetime"].iloc[pos]
        logger.warning("  %s: розрив %s -> %s (%s)", label, gap_start, gap_end, gap_end - gap_start)
    if n_gaps > 10:
        logger.warning("%s: ще %d розривів не показано (обмежено першими 10)", label, n_gaps - 10)


def _check_2020_2021_boundary(df: pd.DataFrame, timeframe: str, label: str) -> None:
    """
    Перевіряє окремо межу 2020-12-31 -> 2021-01-01, де раніше починалась
    історія (до розширення DOWNLOAD_SINCE_DATE назад до 2017-08-17).
    """
    expected_step = EXPECTED_STEP.get(timeframe)
    if expected_step is None:
        return

    window = df[
        (df["datetime"] >= LEGACY_HISTORY_BOUNDARY - pd.Timedelta(days=2))
        & (df["datetime"] <= LEGACY_HISTORY_BOUNDARY + pd.Timedelta(days=2))
    ]
    if window.empty:
        logger.warning(
            "%s: немає рядків біля межі 2020/2021 (%s +-2 дні) — перевір діапазон файлу",
            label, LEGACY_HISTORY_BOUNDARY.date(),
        )
        return

    diffs = window["datetime"].diff().dropna()
    anomalies = diffs[diffs != expected_step]

    if anomalies.empty:
        logger.info(
            "%s: межа 2020/2021 чиста — крок скрізь == %s навколо %s, розривів/аномалій немає",
            label, expected_step, LEGACY_HISTORY_BOUNDARY.date(),
        )
    else:
        logger.warning(
            "%s: АНОМАЛІЯ кроку на межі 2020/2021 — %d випадків відхилення від %s поруч із %s",
            label, len(anomalies), expected_step, LEGACY_HISTORY_BOUNDARY.date(),
        )


def _check_staleness(df: pd.DataFrame, timeframe: str, label: str) -> None:
    """
    Перевіряє, що серія доходить до "сьогодні" (з допуском STALENESS_TOLERANCE).

    Якщо остання дата відстає більше за поріг — це ГУЧНИЙ WARNING про обрив
    серії (делістинг, зупинка торгів тощо), а не тиха звичайна затримка
    оновлення даних. Саме цю перевірку не було раніше, і саме тому обрив
    EOS/USDT (2025-05-26) пройшов непоміченим.
    """
    tolerance = STALENESS_TOLERANCE.get(timeframe)
    expected_step = EXPECTED_STEP.get(timeframe)
    if tolerance is None:
        logger.warning("%s: невідомий поріг застарілості для таймфрейму '%s' — перевірку пропущено", label, timeframe)
        return

    now_utc = pd.Timestamp.now(tz="UTC")
    last_date = df["datetime"].iloc[-1]
    lag = now_utc - last_date

    if lag <= tolerance:
        logger.info("%s: серія актуальна, відставання від сьогодні %s (поріг %s)", label, lag, tolerance)
        return

    missing_bars_str = f"~{int(lag / expected_step)}" if expected_step else "?"
    logger.warning(
        "%s: СЕРІЯ ОБРИВАЄТЬСЯ РАНІШЕ ОЧІКУВАНОГО. Остання дата: %s. Сьогодні: %s. "
        "Відставання: %s (поріг для %s: %s). Приблизно %s пропущених барів. "
        "Схоже на делістинг/зупинку торгів пари на біржі, а НЕ на звичайну затримку "
        "оновлення даних — перевір статус пари на Binance вручну.",
        label, last_date.date(), now_utc.date(), lag, timeframe, tolerance, missing_bars_str,
    )


def _check_first_date_boundary(df: pd.DataFrame, label: str) -> None:
    """
    Перевіряє, що перша дата серії не пізніша за EXPLORATION_END_DATE.

    Якщо монета з'явилась на біржі вже ПІСЛЯ кінця exploration-періоду, у неї
    немає жодного exploration-бару — вся її історія потрапляє в holdout, і
    вона фактично не бере участі в EDA/підборі логіки крос-секційної стратегії.
    """
    first_date = df["datetime"].iloc[0]
    if first_date > LATEST_ACCEPTABLE_FIRST_DATE:
        logger.warning(
            "%s: ПЕРША ДАТА СЕРІЇ (%s) ПІЗНІШЕ МЕЖІ EXPLORATION_END_DATE (%s) — "
            "у монети немає жодного exploration-бару, вся серія потрапляє в holdout. "
            "Непридатна для крос-секційного заміру в exploration-періоді.",
            label, first_date.date(), LATEST_ACCEPTABLE_FIRST_DATE.date(),
        )
    else:
        logger.info(
            "%s: перша дата (%s) не пізніше межі EXPLORATION_END_DATE (%s) — OK",
            label, first_date.date(), LATEST_ACCEPTABLE_FIRST_DATE.date(),
        )


def _report_nan(df: pd.DataFrame, label: str, has_indicators: bool) -> int | None:
    """
    Для сирих даних — просто рахує NaN (їх не повинно бути взагалі).
    Для processed — окремо звітує NaN у перших INDICATOR_WARMUP_BARS рядках
    (очікувано через ema_200 та інші довгі вікна) і окремо ПІСЛЯ прогріву
    (там NaN бути не повинно).

    :return: сумарна кількість NaN-комірок ПІСЛЯ прогріву (0, якщо чисто),
        або None, якщо перевірка "після прогріву" неможлива/незастосовна
        (сирі дані або файл коротший за warm-up-вікно)
    """
    if not has_indicators:
        nan_counts = df.isna().sum()
        nan_nonzero = nan_counts[nan_counts > 0]
        if nan_nonzero.empty:
            logger.info("%s: NaN відсутні", label)
        else:
            logger.warning("%s: знайдено NaN у сирих даних (несподівано):\n%s", label, nan_nonzero.to_string())
        return None

    if len(df) <= INDICATOR_WARMUP_BARS:
        logger.warning(
            "%s: у файлі %d рядків, менше за warm-up-вікно (%d) — перевірку 'після прогріву' пропущено",
            label, len(df), INDICATOR_WARMUP_BARS,
        )
        nan_counts = df.isna().sum()
        nan_nonzero = nan_counts[nan_counts > 0]
        logger.info("%s: NaN по колонках (весь файл):\n%s", label, nan_nonzero.to_string() if not nan_nonzero.empty else "(немає)")
        return None

    warmup_part = df.iloc[:INDICATOR_WARMUP_BARS]
    post_warmup_part = df.iloc[INDICATOR_WARMUP_BARS:]

    nan_warmup = warmup_part.isna().sum()
    nan_warmup_nonzero = nan_warmup[nan_warmup > 0]
    nan_post = post_warmup_part.isna().sum()
    nan_post_nonzero = nan_post[nan_post > 0]
    nan_post_total = int(nan_post.sum())

    logger.info(
        "%s: NaN у перших %d барах (warm-up, очікувано):\n%s",
        label, INDICATOR_WARMUP_BARS,
        nan_warmup_nonzero.to_string() if not nan_warmup_nonzero.empty else "(немає)",
    )

    if nan_post_nonzero.empty:
        logger.info("%s: NaN ПІСЛЯ прогріву (рядки %d+) відсутні — OK", label, INDICATOR_WARMUP_BARS)
    else:
        logger.error(
            "%s: знайдено NaN ПІСЛЯ прогріву (рядки %d+) — це НЕ очікувано, потребує розбору:\n%s",
            label, INDICATOR_WARMUP_BARS, nan_post_nonzero.to_string(),
        )

    return nan_post_total


def verify_file(filepath: Path, timeframe: str, symbol: str, kind: str, has_indicators: bool) -> dict | None:
    """
    Виконує повний набір перевірок для одного файлу і логує результат.

    :param kind: "raw" або "processed" — лише для читабельного лейбла в логах
    :return: для processed-файлів, що успішно пройшли завантаження — словник
        статистики {symbol, first_date, last_date, n_bars, n_nan_post_warmup}
        для подальшої підсумкової таблиці; інакше None
    """
    label = f"{symbol} {timeframe} {kind}"
    df = _load_csv(filepath)
    if df is None:
        return None

    if "datetime" not in df.columns:
        logger.error("%s: відсутня колонка 'datetime' — перевірку пропущено", label)
        return None

    logger.info(
        "%s: %d рядків, період %s -> %s",
        label, len(df), df["datetime"].iloc[0], df["datetime"].iloc[-1],
    )

    n_nan_post_warmup = _report_nan(df, label, has_indicators)
    _check_missing_intervals(df, timeframe, label)
    _check_2020_2021_boundary(df, timeframe, label)
    _check_staleness(df, timeframe, label)
    _check_first_date_boundary(df, label)

    if df["datetime"].iloc[0] <= EARLY_PERIOD_CAUTION_END:
        logger.warning(
            "%s: файл містить дані 2017-2020 — структурно інший ринок (менша "
            "ліквідність, менше інституційних грошей). Дані лишаємо, але це "
            "треба враховувати при інтерпретації статистик з цього періоду.",
            label,
        )

    if not has_indicators:
        return None

    return {
        "symbol": symbol,
        "first_date": df["datetime"].iloc[0].date(),
        "last_date": df["datetime"].iloc[-1].date(),
        "n_bars": len(df),
        "n_nan_post_warmup": n_nan_post_warmup,
    }


def _log_summary_table(rows: list[dict]) -> None:
    """Логує підсумкову таблицю по кожній крос-секційній монеті: дати, бари, NaN після прогріву."""
    if not rows:
        logger.warning("Підсумкова таблиця порожня — немає жодного успішно перевіреного крос-секційного файлу")
        return

    rows_sorted = sorted(rows, key=lambda r: r["first_date"])

    header = f"{'Символ':<10} {'Перша дата':<12} {'Остання дата':<12} {'Барів':>7} {'NaN після прогріву':>20}"
    separator = "-" * len(header)
    lines = [header, separator]
    for row in rows_sorted:
        nan_str = str(row["n_nan_post_warmup"]) if row["n_nan_post_warmup"] is not None else "н/д"
        lines.append(
            f"{row['symbol']:<10} {row['first_date'].isoformat():<12} {row['last_date'].isoformat():<12} "
            f"{row['n_bars']:>7} {nan_str:>20}"
        )

    logger.info("ПІДСУМКОВА ТАБЛИЦЯ (крос-секційний %s-набір, %d монет):\n%s", CROSS_SECTIONAL_TIMEFRAME, len(rows), "\n".join(lines))


def _log_yearly_coverage(rows: list[dict]) -> None:
    """
    Логує кількість доступних монет по кожному року (COVERAGE_YEAR_START..
    COVERAGE_YEAR_END) — монета вважається "доступною" в році Y, якщо її
    серія перетинається з цим роком (first_date.year <= Y <= last_date.year).
    Потрібно для оцінки статистичної потужності крос-секційного заміру
    (розмір терциля ~ N/3 монет доступних у році).
    """
    if not rows:
        return

    lines = []
    for year in range(COVERAGE_YEAR_START, COVERAGE_YEAR_END + 1):
        available = sorted(
            row["symbol"] for row in rows
            if row["first_date"].year <= year <= row["last_date"].year
        )
        n = len(available)
        tercile_size = n // 3
        lines.append(
            f"{year}: {n} монет (терциль ~{tercile_size}) — {', '.join(available) if available else '(немає)'}"
        )

    logger.info(
        "ПОКРИТТЯ ЮНІВЕРСУ ПО РОКАХ (%d-%d):\n%s",
        COVERAGE_YEAR_START, COVERAGE_YEAR_END, "\n".join(lines),
    )


def main() -> None:
    """Перевіряє raw і processed файли для всіх пар (symbol, timeframe) з config.PIPELINE_TARGETS."""
    logger.warning(
        "SURVIVORSHIP BIAS (нагадування): CROSS_SECTIONAL_SYMBOLS (%s) складено з "
        "монет, що торгуються СЬОГОДНІ (крім EOS/USDT, яку навмисно лишили попри "
        "обрив серії — див. STALENESS нижче). Монети, що зникли/втратили лістинг "
        "Binance за 2018-2023 і НЕ повернулись, у наборі відсутні — це систематично "
        "завищує результат крос-секційної стратегії. Враховуй це при інтерпретації "
        "результатів кроку 7.9.4.",
        ", ".join(CROSS_SECTIONAL_SYMBOLS),
    )

    summary_rows: list[dict] = []

    for symbol, timeframe in PIPELINE_TARGETS:
        filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"

        logger.info("=== %s / %s (raw) ===", symbol, timeframe)
        verify_file(RAW_DATA_DIR / filename, timeframe, symbol, "raw", has_indicators=False)

        logger.info("=== %s / %s (processed) ===", symbol, timeframe)
        stats = verify_file(PROCESSED_DATA_DIR / filename, timeframe, symbol, "processed", has_indicators=True)

        if stats is not None and timeframe == CROSS_SECTIONAL_TIMEFRAME and symbol in CROSS_SECTIONAL_SYMBOLS:
            summary_rows.append(stats)

    _log_summary_table(summary_rows)
    _log_yearly_coverage(summary_rows)


if __name__ == "__main__":
    main()
