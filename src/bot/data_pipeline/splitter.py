"""
Модуль розділення processed-файлу (з усіма індикаторами, порахованими на
повній серії) на exploration/holdout за часом.

Це НЕ crossvalidation-спліт і не випадковий shuffle: часовий ряд лишається
впорядкованим, розділення відбувається строго по даті свічки. Holdout —
дані, які НЕ використовуються під час EDA і підбору логіки стратегії,
лише для фінальної одноразової перевірки (запобігання overfitting і
lookahead bias — див. docs/trading_decisions.md).

Індикатори тут НЕ перераховуються — вони вже пораховані на повній серії
в indicators.py. Цей модуль лише фільтрує вже готові рядки за датою.
"""

from __future__ import annotations

import pandas as pd

from bot.utils.logger import get_logger

logger = get_logger(__name__)


def split_exploration_holdout(
    df: pd.DataFrame,
    exploration_end_date: str,
    holdout_start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ділить processed DataFrame на exploration і holdout за датою свічки.

    Exploration: усі рядки з датою <= exploration_end_date (включно, до кінця доби).
    Holdout:     усі рядки з датою >= holdout_start_date (включно, з початку доби).

    Порядок рядків усередині кожної частини зберігається — жодного shuffle,
    це часовий ряд.

    :param df: processed DataFrame з колонкою "datetime" (tz-aware UTC, як
               зберігають downloader.py / indicators.py)
    :param exploration_end_date: остання дата exploration-періоду, "YYYY-MM-DD"
    :param holdout_start_date: перша дата holdout-періоду, "YYYY-MM-DD"
    :return: (exploration_df, holdout_df) — копії з відновленим RangeIndex
    :raises ValueError: якщо відсутня колонка "datetime", або якщо сума рядків
        exploration + holdout не дорівнює загальній кількості рядків вхідного df
        (ознака пропуску або перетину між межами дат — жоден рядок не повинен
        загубитися чи задублюватися)
    """
    if "datetime" not in df.columns:
        raise ValueError("У вхідному DataFrame відсутня колонка 'datetime'")

    datetime_col = pd.to_datetime(df["datetime"], utc=True)
    exploration_end_midnight = pd.Timestamp(exploration_end_date, tz="UTC")
    holdout_start = pd.Timestamp(holdout_start_date, tz="UTC")

    # "включно" по ДАТІ, а не по голому timestamp 00:00 — інакше свічки, що
    # припадають на середину exploration_end_date (крім рівно опівночі),
    # помилково відсіклися б від exploration.
    exploration_end_inclusive = (
        exploration_end_midnight + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    )

    exploration_mask = datetime_col <= exploration_end_inclusive
    holdout_mask = datetime_col >= holdout_start

    exploration_df = df.loc[exploration_mask].reset_index(drop=True)
    holdout_df = df.loc[holdout_mask].reset_index(drop=True)

    total_input_rows = len(df)
    total_output_rows = len(exploration_df) + len(holdout_df)

    if total_output_rows != total_input_rows:
        problem = "перетин" if total_output_rows > total_input_rows else "пропуск рядків"
        raise ValueError(
            f"Розбиття не пройшло перевірку цілісності: "
            f"exploration ({len(exploration_df)}) + holdout ({len(holdout_df)}) = "
            f"{total_output_rows} рядків, а у вхідному файлі — {total_input_rows}. "
            f"Схоже на {problem} між EXPLORATION_END_DATE ({exploration_end_date}) "
            f"і HOLDOUT_START_DATE ({holdout_start_date}) — перевір ці межі в config/settings.py."
        )

    logger.info(
        "Перевірка цілісності пройдена: %d (вхід) = %d (exploration) + %d (holdout)",
        total_input_rows, len(exploration_df), len(holdout_df),
    )

    return exploration_df, holdout_df


def log_split_summary(name: str, df: pd.DataFrame) -> None:
    """
    Логує самоперевірку одного зі спліт-наборів: кількість рядків, перша й
    остання дата, кількість NaN по кожній колонці (де вона є).

    Використовується для власної перевірки користувачем — вміст holdout не
    призначений для показу/обговорення в чаті.

    :param name: людяна назва для логів, напр. "exploration" або "holdout"
    :param df: DataFrame одного зі спліт-файлів (з колонкою "datetime")
    """
    if df.empty:
        logger.warning(
            "%s: 0 рядків — перевір межі EXPLORATION_END_DATE/HOLDOUT_START_DATE "
            "у config/settings.py відносно фактичного діапазону даних",
            name,
        )
        return

    first_date = df["datetime"].iloc[0]
    last_date = df["datetime"].iloc[-1]
    nan_counts = df.isna().sum()
    nan_counts_nonzero = nan_counts[nan_counts > 0]

    logger.info("%s: %d рядків, період %s -> %s", name, len(df), first_date, last_date)

    if nan_counts_nonzero.empty:
        logger.info("%s: NaN по колонках відсутні", name)
    else:
        logger.info("%s: NaN по колонках:\n%s", name, nan_counts_nonzero.to_string())