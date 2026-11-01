"""
Модуль збирання крос-секційного long-формату (крок 7.9.4) з уже готових
per-symbol exploration-файлів (data/processed/{symbol}_{timeframe}_exploration.csv,
вихід splitter.py / split_data.py).

Індикатори тут НЕ потрібні й НЕ переносяться — крос-секційний замір
(momentum за close, ранжування між монетами) використовує лише сирі OHLCV.

Модуль лише читає готові файли, вибирає потрібні колонки, фільтрує за
періодом, конкатенує монети в один long-формат (одна монета × одна дата =
один рядок) і сортує. Жодних перерахунків цін, ресемплу чи нормалізації.

КРИТИЧНО: жодного заповнення пропусків (forward-fill, нулі тощо). Монета,
якої немає в конкретний день (ще не лістована або вже делістована), просто
відсутня в рядках цього дня — інакше вона потрапила б у ранжування з
фіктивною ціною, що спотворює крос-секційний замір.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bot.utils.logger import get_logger

logger = get_logger(__name__)

# Колонки, які переносимо в long-формат — без індикаторів.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _exploration_filepath(symbol: str, timeframe: str, processed_data_dir: Path) -> Path:
    """
    Шлях до exploration-файлу однієї монети — та сама схема іменування,
    що й у split_data.py: базова валюта в нижньому регістрі + timeframe.
    Напр. "BTC/USDT" + "1d" -> "btc_1d_exploration.csv".
    """
    base_currency = symbol.split("/")[0].lower()
    return processed_data_dir / f"{base_currency}_{timeframe}_exploration.csv"


def _load_symbol_ohlcv(
    symbol: str,
    timeframe: str,
    processed_data_dir: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """
    Завантажує exploration-файл однієї монети, лишає лише OHLCV-колонки й
    рядки в межах [start_date, end_date] (включно, по календарній даті),
    додає колонку symbol.

    Файл, якого немає, або в якого немає жодного рядка в заданому періоді —
    НЕ помилка процесу: логуємо попередження і монету просто пропускаємо
    (вона природно відсутня в результуючому long-формі за цей період).

    :return: DataFrame [date, symbol, open, high, low, close, volume], або
        None, якщо монету для цього періоду взяти не вдалось
    """
    filepath = _exploration_filepath(symbol, timeframe, processed_data_dir)
    if not filepath.exists():
        logger.warning("%s: exploration-файл не знайдено (%s) — монету пропущено", symbol, filepath)
        return None

    df = pd.read_csv(filepath)
    if "datetime" not in df.columns:
        logger.warning("%s: у файлі немає колонки 'datetime' — монету пропущено", symbol)
        return None

    missing_cols = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.warning("%s: відсутні колонки %s — монету пропущено", symbol, missing_cols)
        return None

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    mask = (df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)
    df = df.loc[mask]

    if df.empty:
        logger.warning(
            "%s: немає рядків у періоді %s -> %s — монету пропущено для цього періоду",
            symbol, start_date, end_date,
        )
        return None

    result = df[["datetime", *OHLCV_COLUMNS]].copy()
    result = result.rename(columns={"datetime": "date"})
    # Денні бари завжди опівночі UTC — .dt.date не втрачає інформацію,
    # лише прибирає надлишковий часовий компонент.
    result["date"] = result["date"].dt.date
    result["symbol"] = symbol

    return result[["date", "symbol", *OHLCV_COLUMNS]]


def build_cross_sectional_long(
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    processed_data_dir: Path,
) -> pd.DataFrame:
    """
    Збирає long-формат крос-секційного датасету: одна монета × одна дата =
    один рядок, колонки [date, symbol, open, high, low, close, volume].

    Джерело — вже готові *_exploration.csv per-symbol файли (вихід
    split_data.py). Індикатори не переносяться (не потрібні для
    крос-секційного заміру на close). Монети, яких немає в конкретний день
    (ще не лістовані або вже делістовані — напр. EOS/WAVES поза межами їхньої
    активної історії), просто відсутні в рядках цього дня, БЕЗ forward-fill
    чи заповнення нулями.

    :param symbols: список монет (ccxt-формат, напр. "BTC/USDT")
    :param timeframe: таймфрейм (для крос-секційного заміру — "1d")
    :param start_date: початок періоду, включно, "YYYY-MM-DD"
    :param end_date: кінець періоду, включно, "YYYY-MM-DD"
    :param processed_data_dir: каталог data/processed/
    :return: DataFrame, відсортований за date, потім symbol
    :raises ValueError: якщо жодної монети не вдалось завантажити за період
    """
    frames: list[pd.DataFrame] = []
    skipped_symbols: list[str] = []

    for symbol in symbols:
        symbol_df = _load_symbol_ohlcv(symbol, timeframe, processed_data_dir, start_date, end_date)
        if symbol_df is None:
            skipped_symbols.append(symbol)
            continue
        frames.append(symbol_df)

    if not frames:
        raise ValueError(
            f"Жодної монети не вдалося завантажити для періоду {start_date} -> {end_date}"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)

    logger.info(
        "Зібрано крос-секційний long-формат (%s -> %s): %d рядків, %d монет включено%s",
        start_date, end_date, len(combined), len(frames),
        f", {len(skipped_symbols)} пропущено ({', '.join(skipped_symbols)})" if skipped_symbols else "",
    )

    return combined


def log_cross_sectional_summary(df: pd.DataFrame, label: str, min_coins_per_day: int = 15) -> None:
    """
    Логує підсумкову статистику long-формату: кількість рядків, унікальних
    дат, кількість монет у перший і останній день періоду, і перевірку — чи
    кожен день має щонайменше min_coins_per_day монет (інакше перелічує
    конкретні дати, де монет менше).
    """
    if df.empty:
        logger.warning("%s: датасет порожній — 0 рядків", label)
        return

    n_rows = len(df)
    unique_dates = sorted(df["date"].unique())
    n_dates = len(unique_dates)
    first_date, last_date = unique_dates[0], unique_dates[-1]

    n_coins_first = df.loc[df["date"] == first_date, "symbol"].nunique()
    n_coins_last = df.loc[df["date"] == last_date, "symbol"].nunique()

    logger.info(
        "%s: %d рядків, %d унікальних дат (%s -> %s). Монет у перший день (%s): %d. "
        "Монет в останній день (%s): %d.",
        label, n_rows, n_dates, first_date, last_date, first_date, n_coins_first, last_date, n_coins_last,
    )

    coins_per_day = df.groupby("date")["symbol"].nunique()
    thin_days = coins_per_day[coins_per_day < min_coins_per_day]

    if thin_days.empty:
        logger.info("%s: у кожен із %d днів щонайменше %d монет — OK", label, n_dates, min_coins_per_day)
    else:
        logger.warning(
            "%s: %d днів з менше ніж %d монетами (дата: кількість монет):\n%s",
            label, len(thin_days), min_coins_per_day, thin_days.to_string(),
        )
