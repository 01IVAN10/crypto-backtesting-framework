"""
Модуль розрахунку технічних індикаторів та статистик для EDA (профілювання ринку).

Важливо: індикатори тут — НЕ сигнали стратегії. Це дескриптивний матеріал для
дослідження характеру ринку (волатильність, сила тренду, режими). Логіка
входу/виходу конкретної стратегії живе окремо в src/bot/strategies/.

Усі індикатори розраховуються власними засобами на pandas (ewm, rolling),
без сторонніх TA-бібліотек (pandas-ta тощо), на ПОВНІЙ вхідній серії.

Розділення на exploration/holdout — окремий наступний крок, що виконується
ПІСЛЯ цього модуля. Через це на початку кожної серії будуть NaN (warm-up
період rolling/ewm-вікон) — це очікувана поведінка, а не помилка.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Експоненційна ковзна середня (Exponential Moving Average) з вказаним span."""
    return series.ewm(span=span, adjust=False).mean()


def _wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
    """
    Згладжування Уайлдера (Wilder's smoothing) — еквівалент ewm з alpha = 1/period.
    Класична основа формул ATR, RSI, ADX (Welles Wilder, "New Concepts in
    Technical Trading Systems", 1978).
    """
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range — максимум із трьох варіантів діапазону свічки, що враховує
    гепи відносно попереднього закриття:
        1) high - low
        2) |high - prev_close|
        3) |low - prev_close|
    """
    prev_close = df["close"].shift(1)
    range_hl = df["high"] - df["low"]
    range_hc = (df["high"] - prev_close).abs()
    range_lc = (df["low"] - prev_close).abs()
    return pd.concat([range_hl, range_hc, range_lc], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — True Range, згладжений методом Уайлдера."""
    true_range = _true_range(df)
    return _wilder_smoothing(true_range, period)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index за класичною формулою Уайлдера:
        RS  = середній приріст / середній збиток (обидва згладжені Уайлдером)
        RSI = 100 - 100 / (1 + RS)

    Коли середній збиток дорівнює нулю (безперервне зростання ціни),
    RSI коректно прирівнюється до 100 замість NaN/inf від ділення на нуль.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = _wilder_smoothing(gain, period)
    avg_loss = _wilder_smoothing(loss, period)

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def _adx(df: pd.DataFrame, atr: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index за Уайлдером — сила тренду незалежно від напрямку
    (0-100, вище означає сильніший тренд; сам ADX НЕ вказує напрямок).

    Кроки:
        +DM/-DM (напрямковий рух)
        -> згладжування Уайлдера
        -> +DI/-DI (нормалізовані на ATR)
        -> DX = 100 * |+DI - -DI| / (+DI + -DI)
        -> ADX = DX, згладжений Уайлдером

    :param atr: попередньо розрахований ATR (щоб не рахувати True Range двічі)
    """
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    smoothed_plus_dm = _wilder_smoothing(plus_dm, period)
    smoothed_minus_dm = _wilder_smoothing(minus_dm, period)

    plus_di = 100.0 * (smoothed_plus_dm / atr)
    minus_di = 100.0 * (smoothed_minus_dm / atr)

    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    # Коли +DI і -DI обидва нульові (немає руху ціни), di_sum = 0 -> DX невизначений, беремо 0
    dx = dx.where(di_sum != 0, 0.0)

    adx = _wilder_smoothing(dx, period)
    return adx


def _bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Смуги Боллінджера: SMA(period) +- num_std стандартних відхилень (population std, ddof=0).

    :return: (верхня межа, нижня межа, ширина = (верхня - нижня) / SMA)
    """
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)

    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower) / sma

    return upper, lower, width


def _log_returns(close: pd.Series) -> pd.Series:
    """Логарифмічна дохідність між сусідніми барами: ln(close_t / close_{t-1})."""
    return np.log(close / close.shift(1))


def _realized_volatility(log_returns: pd.Series, window: int = 20) -> pd.Series:
    """Реалізована волатильність — rolling-стандартне відхилення log returns."""
    return log_returns.rolling(window=window).std()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає до сирого OHLCV набір індикаторів та статистик для EDA-профілювання ринку.

    Індикатори НЕ є сигналами стратегії — призначені для дослідження характеру
    ринку (тренд, волатильність, режими). Рахуються на всій вхідній серії без
    розділення на exploration/holdout (це окремий наступний крок і виконується
    поза цим модулем).

    Додані колонки:
        ema_20, ema_50, ema_200   — експоненційні ковзні середні
        atr_14                    — Average True Range
        rsi_14                    — Relative Strength Index
        adx_14                    — Average Directional Index (сила тренду)
        bb_upper, bb_lower, bb_width — смуги Боллінджера (20, 2σ) і їх ширина
        log_return                — логарифмічна дохідність бар-до-бару
        realized_vol_20           — rolling std log returns, вікно 20

    :param df: DataFrame з колонками щонайменше ["open", "high", "low", "close", "volume"],
               відсортований за часом за зростанням (як повертає downloader.py)
    :return: НОВИЙ DataFrame (копія) з доданими колонками; вхідний df не мутується
    :raises ValueError: якщо у df відсутня хоча б одна з обов'язкових колонок
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"У вхідному DataFrame відсутні колонки: {missing}")

    result = df.copy()

    result["ema_20"] = _ema(result["close"], 20)
    result["ema_50"] = _ema(result["close"], 50)
    result["ema_200"] = _ema(result["close"], 200)

    atr = _atr(result, 14)
    result["atr_14"] = atr
    result["rsi_14"] = _rsi(result["close"], 14)
    result["adx_14"] = _adx(result, atr, 14)

    bb_upper, bb_lower, bb_width = _bollinger_bands(result["close"], period=20, num_std=2.0)
    result["bb_upper"] = bb_upper
    result["bb_lower"] = bb_lower
    result["bb_width"] = bb_width

    result["log_return"] = _log_returns(result["close"])
    result["realized_vol_20"] = _realized_volatility(result["log_return"], window=20)

    added_columns = [
        "ema_20", "ema_50", "ema_200", "atr_14", "rsi_14", "adx_14",
        "bb_upper", "bb_lower", "bb_width", "log_return", "realized_vol_20",
    ]
    n_warmup_rows = result[added_columns].isna().any(axis=1).sum()
    logger.info(
        "Розраховано індикатори: %d рядків, %d колонок (з них %d рядків із NaN "
        "у якомусь із індикаторів через warm-up-період rolling/ewm-вікон — "
        "найдовше вікно тут rolling(20) для Bollinger/realized_vol, це очікувано, не помилка)",
        len(result), result.shape[1], n_warmup_rows,
    )

    return result