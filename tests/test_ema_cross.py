"""
tests/test_ema_cross.py

Тести для src/bot/strategies/ema_cross.py.

Ключовий сценарій (за завданням): перетин EMA(20)/EMA(50) знизу вгору сам
собою НЕ повинен давати сигнал на вхід, якщо close на цьому барі нижче
EMA(200) — трендовий фільтр має заблокувати сигнал.

Додатково: сигнал на вхід, коли фільтр тренду виконується; сигнал на вихід
(cross_down), який НЕ залежить від EMA(200); помилка при відсутніх колонках.
"""

import pandas as pd
import pytest

from bot.strategies.ema_cross import EMACrossStrategy


def _make_df() -> pd.DataFrame:
    """
    Мінімальний сконструйований DataFrame на 4 барах (індекс — час умовно):

    Бар 0: розігрів (перший бар, shift(1) дає NaN) -> сигналу немає.
    Бар 1: EMA-20 перетинає EMA-50 знизу вгору, АЛЕ close(50) < EMA-200(100)
           -> трендовий фільтр блокує вхід, очікуємо signal=0 (НЕ 1).
    Бар 2: EMA-20 перетинає EMA-50 згори вниз (зворотний сигнал) -> -1,
           незалежно від співвідношення close/EMA-200.
    Бар 3: EMA-20 знову перетинає EMA-50 знизу вгору, close(150) > EMA-200(100)
           -> трендовий фільтр виконується, очікуємо signal=1 (sanity-перевірка,
           що фільтр не блокує сигнали завжди).
    """
    return pd.DataFrame(
        {
            "close":   [100.0, 50.0, 100.0, 150.0],
            "ema_20":  [1.0, 3.0, 1.0, 3.0],
            "ema_50":  [2.0, 2.0, 2.0, 2.0],
            "ema_200": [100.0, 100.0, 100.0, 100.0],
        }
    )


def test_no_entry_signal_when_close_below_ema_long_despite_cross() -> None:
    """
    Головний сценарій із завдання: перетин 20/50 знизу вгору є, але
    close нижче EMA(200) -> сигналу на вхід (1) бути НЕ повинно.
    """
    df = _make_df()
    strategy = EMACrossStrategy()

    result = strategy.generate_signals(df)

    # Бар 1: cross_up є (перевіряємо це окремо нижче), але close < ema_200.
    assert result.loc[1, "signal"] == 0
    assert result.loc[1, "signal"] != 1

    # Перевіряємо саму передумову тесту: перетин на барі 1 дійсно стався
    # (інакше тест нічого не доводить — він мав би пройти і без фільтра).
    assert df.loc[0, "ema_20"] <= df.loc[0, "ema_50"]  # prev: short <= medium
    assert df.loc[1, "ema_20"] > df.loc[1, "ema_50"]   # now: short > medium


def test_entry_signal_when_trend_filter_confirmed() -> None:
    """Sanity: коли close вище EMA(200), той самий тип перетину дає сигнал 1."""
    df = _make_df()
    strategy = EMACrossStrategy()

    result = strategy.generate_signals(df)

    assert result.loc[3, "signal"] == 1


def test_reverse_exit_signal_ignores_trend_filter() -> None:
    """
    Зворотний сигнал (cross_down) не залежить від EMA(200) — вихід повинен
    спрацьовувати завжди, навіть якщо на цьому барі close вище EMA(200).
    """
    df = _make_df()
    strategy = EMACrossStrategy()

    result = strategy.generate_signals(df)

    assert result.loc[2, "signal"] == -1


def test_first_bar_has_no_signal() -> None:
    """Перший бар: shift(1) дає NaN -> порівняння False -> сигналу немає."""
    df = _make_df()
    strategy = EMACrossStrategy()

    result = strategy.generate_signals(df)

    assert result.loc[0, "signal"] == 0


def test_missing_required_column_raises_keyerror() -> None:
    """Відсутність обов'язкової колонки (напр. ema_200) має явно ламати виконання."""
    df = _make_df().drop(columns=["ema_200"])
    strategy = EMACrossStrategy()

    with pytest.raises(KeyError):
        strategy.generate_signals(df)


def test_custom_periods_use_matching_column_names() -> None:
    """Стратегія з нестандартними періодами читає колонки за відповідними назвами."""
    df = pd.DataFrame(
        {
            "close":  [100.0, 150.0],
            "ema_5":  [1.0, 3.0],
            "ema_10": [2.0, 2.0],
            "ema_30": [100.0, 100.0],
        }
    )
    strategy = EMACrossStrategy(ema_short_period=5, ema_medium_period=10, ema_long_period=30)

    result = strategy.generate_signals(df)

    assert result.loc[1, "signal"] == 1


def test_invalid_period_order_raises_value_error() -> None:
    """Періоди мають бути впорядковані short < medium < long."""
    with pytest.raises(ValueError):
        EMACrossStrategy(ema_short_period=50, ema_medium_period=20, ema_long_period=200)


def test_does_not_mutate_input_dataframe() -> None:
    """generate_signals не повинна змінювати вхідний df на місці (повертає копію)."""
    df = _make_df()
    original_columns = list(df.columns)
    strategy = EMACrossStrategy()

    strategy.generate_signals(df)

    assert list(df.columns) == original_columns
    assert "signal" not in df.columns