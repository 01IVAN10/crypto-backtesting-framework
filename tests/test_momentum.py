"""
tests/test_momentum.py

Тести для src/bot/strategies/momentum.py.

Два незалежні сценарії на lookahead (за завданням, зумисно розділені —
кожен ловить РІЗНИЙ клас помилки):

1. Поріг верхнього терциля для бару t НЕ повинен використовувати бари
   ПІСЛЯ t (класичний lookahead — дані з майбутнього). Ловлять:
   test_tercile_threshold_ignores_future_data_no_lookahead,
   test_truncated_dataframe_gives_identical_past_results.

2. Поріг для бару t НЕ повинен включати сам бар t у власне референсне
   вікно (self-inclusion) — вікно має бути [t-tercile_window, t-1], а не
   [t-tercile_window+1, t]. Це НЕ lookahead у класичному сенсі (жодних
   барів з майбутнього), але порушує вимогу "поріг рахується лише з
   минулих відносно t даних": значення бере участь у визначенні порогу,
   який його ж класифікує. Сценарій (1) цього НЕ ловить: обрізання/мутація
   даних ПІСЛЯ t не чіпає сам бар t, який в обох випадках лишається у
   власному вікні. Ловить: test_threshold_matches_reference_formula_excluding_current_bar,
   test_threshold_hand_traceable_small_window_example.

Додатково: test_momentum_return_matches_reference_close_to_close_formula
перевіряє незалежною формулою, що 30-денна дохідність на барі t рахується
рівно з close[t] і close[t-30] — обидва вже відомі на закритті бару t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.strategies.momentum import Momentum30Strategy


def _make_synthetic_close(n: int) -> np.ndarray:
    """
    Детермінований (без RNG) синтетичний ряд close: висхідний тренд +
    дві накладені хвилі різного періоду — дає достатньо варіації
    30-денного моментуму, щоб стан активний/неактивний реально змінювався,
    без залежності тесту від конкретного seed генератора випадкових чисел.
    """
    idx = np.arange(n, dtype=float)
    trend = 100.0 * (1.0 + 0.0015 * idx)
    wave = 1.0 + 0.08 * np.sin(idx / 9.0) + 0.05 * np.sin(idx / 37.0 + 1.0)
    return trend * wave


def _make_df(close: np.ndarray, start: str = "2018-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(close), freq="1D", tz="UTC")
    return pd.DataFrame({"datetime": dates, "close": close})


# Перший валідний поріг з'являється на індексі (lookback + tercile_window):
# поріг рахується з momentum_return.shift(1) (виключає бар t), а перші
# non-NaN momentum_return з'являються з індексу lookback. Зсунутій серії
# потрібно tercile_window НЕ-NaN спостережень У ВІКНІ, ЩО ЗАКІНЧУЄТЬСЯ на
# t-1, тобто перший валідний поріг — на t = lookback + tercile_window.
# До цього індексу включно (WARMUP рядків, індекси 0..WARMUP-1) поріг/сигнал
# мають бути відсутні (NaN/0).
WARMUP = 30 + 252


def test_tercile_threshold_ignores_future_data_no_lookahead() -> None:
    """
    Головний тест на lookahead: драматична зміна барів ПІСЛЯ точки
    відсічення не повинна впливати на momentum_return, поріг, стан чи
    сигнал ДО і НА цій точці. Якби поріг рахувався по всій вибірці
    (а не ковзним вікном з минулого), ця зміна майбутнього посунула б
    розподіл і, отже, поріг заднім числом.
    """
    n = 400
    cutoff = 350
    close_a = _make_synthetic_close(n)
    strategy = Momentum30Strategy()

    result_a = strategy.generate_signals(_make_df(close_a.copy()))

    close_b = close_a.copy()
    # Драматичний шок майбутніх барів: зростаючий множник від ×2 до ×10
    # на весь "хвіст" після cutoff.
    close_b[cutoff + 1:] = close_b[cutoff + 1:] * np.linspace(2.0, 10.0, n - cutoff - 1)
    result_b = strategy.generate_signals(_make_df(close_b))

    compare_cols = [
        "momentum_return",
        "momentum_tercile_threshold",
        "momentum_state_active",
        "signal",
    ]
    for col in compare_cols:
        pd.testing.assert_series_equal(
            result_a.loc[: cutoff, col].reset_index(drop=True),
            result_b.loc[: cutoff, col].reset_index(drop=True),
            check_names=False,
        )

    # Санітарна перевірка: мутація майбутнього дійсно щось змінила ПІСЛЯ
    # cutoff — інакше тест вище нічого б не доводив (порівнювали б два
    # ідентичні прогони).
    assert not result_a["momentum_return"].iloc[cutoff + 1:].reset_index(drop=True).equals(
        result_b["momentum_return"].iloc[cutoff + 1:].reset_index(drop=True)
    )


def test_truncated_dataframe_gives_identical_past_results() -> None:
    """
    Другорядне підтвердження того самого: прогін на df, обрізаному на
    точці cutoff, дає ті самі значення для всіх барів <= cutoff, що й
    прогін на повному df — незалежно від того, скільки барів "майбутнього"
    взагалі існує.
    """
    n = 400
    cutoff = 350
    close = _make_synthetic_close(n)
    strategy = Momentum30Strategy()

    result_full = strategy.generate_signals(_make_df(close.copy()))
    result_truncated = strategy.generate_signals(_make_df(close[: cutoff + 1].copy()))

    for col in ["momentum_return", "momentum_tercile_threshold", "momentum_state_active", "signal"]:
        pd.testing.assert_series_equal(
            result_full.loc[: cutoff, col].reset_index(drop=True),
            result_truncated[col].reset_index(drop=True),
            check_names=False,
        )


def test_momentum_return_matches_reference_close_to_close_formula() -> None:
    """
    Вимога 2: 30-денна (lookback-денна) log-дохідність на барі t має
    рахуватись рівно як ln(close[t] / close[t-lookback]) — обидва значення
    вже відомі на закритті бару t, жодних інших барів не використовується.

    Перевірка незалежною формулою (не викликом самого strategy._... —
    буквально ln(close/close.shift(lookback)) поверх сирого df), а не
    просто повторним читанням коду стратегії.
    """
    n = 200
    lookback = 30
    close = _make_synthetic_close(n)
    df = _make_df(close)
    strategy = Momentum30Strategy(lookback=lookback)

    result = strategy.generate_signals(df)

    close_series = pd.Series(close)
    expected = np.log(close_series / close_series.shift(lookback))

    pd.testing.assert_series_equal(
        result["momentum_return"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )

    # Явна точкова перевірка на конкретному барі (без покладання лише на
    # векторизовану формулу вище): бар t=100 має дохідність рівно
    # ln(close[100] / close[70]), і НІЯКОГО іншого бару.
    t = 100
    assert result.loc[t, "momentum_return"] == pytest.approx(
        np.log(close[t] / close[t - lookback])
    )


def test_threshold_matches_reference_formula_excluding_current_bar() -> None:
    """
    Вимога 1 (головна перевірка з цього запиту): поріг для бару t має
    рахуватись з БАРІВ [t - tercile_window, t - 1] — БЕЗ momentum_return[t]
    самого бару t. Референсна формула тут написана НЕЗАЛЕЖНО від коду
    стратегії (momentum_return.shift(1).rolling(...)), а не просто
    продубльований рядок з generate_signals — вона буквально виражає
    "quantile по tercile_window точках, що передують t".

    Якби вікно помилково включало t (варіант БЕЗ .shift(1) перед .rolling()),
    цей тест впав би, бо порівнював би дві РІЗНІ серії значень.
    """
    n = 500
    lookback = 30
    tercile_window = 252
    tercile_pct = 0.667
    close = _make_synthetic_close(n)
    df = _make_df(close)
    strategy = Momentum30Strategy(
        lookback=lookback, tercile_window=tercile_window, tercile_pct=tercile_pct
    )

    result = strategy.generate_signals(df)

    close_series = pd.Series(close)
    momentum_return = np.log(close_series / close_series.shift(lookback))
    # Референс: поріг на барі t = quantile() по momentum_return[t-252 .. t-1].
    expected_threshold = (
        momentum_return.shift(1).rolling(window=tercile_window, min_periods=tercile_window).quantile(tercile_pct)
    )

    pd.testing.assert_series_equal(
        result["momentum_tercile_threshold"].reset_index(drop=True),
        expected_threshold.reset_index(drop=True),
        check_names=False,
    )

    # Негативний контроль: поріг НЕ повинен збігатись з "неправильною" версією,
    # де вікно включає сам бар t (без shift(1)) — інакше цей тест нічого не
    # довів би (обидві формули випадково збіглися б). На достатньо мінливому
    # ряді (є варіація momentum_return, wave-компонента в _make_synthetic_close)
    # включення/виключення t-ої точки з 252-точкового вікна для quantile(0.667)
    # хоч десь та й дасть інше число.
    wrong_threshold_including_t = momentum_return.rolling(
        window=tercile_window, min_periods=tercile_window
    ).quantile(tercile_pct)
    valid = result["momentum_tercile_threshold"].notna() & wrong_threshold_including_t.notna()
    assert not np.allclose(
        result.loc[valid, "momentum_tercile_threshold"].to_numpy(),
        wrong_threshold_including_t[valid].to_numpy(),
    )


def test_threshold_hand_traceable_small_window_example() -> None:
    """
    Той самий факт (вимога 1), що й у test_threshold_matches_reference_formula_excluding_current_bar,
    але на КРихітному вікні (tercile_window=5), де значення, що формують
    поріг, можна перелічити вручну й перевірити очима, а не довіряти лише
    векторизованій формулі-референсу.
    """
    lookback = 1
    tercile_window = 5
    tercile_pct = 0.667

    # close підібраний так, щоб momentum_return (log-дохідність за 1 бар)
    # монотонно зростала — тоді вікна й очікувані квантилі легко порахувати вручну.
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 200.0, 108.0])
    df = _make_df(close)
    strategy = Momentum30Strategy(
        lookback=lookback, tercile_window=tercile_window, tercile_pct=tercile_pct
    )

    result = strategy.generate_signals(df)

    momentum_return = np.log(pd.Series(close) / pd.Series(close).shift(lookback))
    # momentum_return[1..9] — валідні (lookback=1). Перший поріг: t такий, що
    # у вікні [t-5, t-1] є 5 валідних momentum_return -> t-5 >= 1 -> t >= 6.
    t = 6
    window_values = momentum_return.iloc[t - tercile_window: t]  # індекси [1..5]
    expected_at_t = window_values.quantile(tercile_pct)

    assert result.loc[t, "momentum_tercile_threshold"] == pytest.approx(expected_at_t)
    # І явно: бар t=6 (сам momentum_return[6], що відповідає стрибку 106->107,
    # НЕ входить у вікно [1..5], яке формує його ж поріг).
    assert t not in window_values.index

    # Наступний бар t=7: аномальний стрибок close[8]=200.0 ЩЕ не потрапив у
    # жодне вікно (з'явиться в momentum_return лише на індексі 8), тож на
    # барі t=7 нічого несподіваного не станеться — перевіряємо лише, що
    # вікно порогу досі [2..6] (без t=7).
    window_values_7 = momentum_return.iloc[7 - tercile_window: 7]
    assert 7 not in window_values_7.index
    assert result.loc[7, "momentum_tercile_threshold"] == pytest.approx(
        window_values_7.quantile(tercile_pct)
    )


def test_no_signal_during_warmup_period() -> None:
    """До набору повного вікна (lookback + tercile_window барів) сигналів бути не повинно."""
    n = 500
    close = _make_synthetic_close(n)
    strategy = Momentum30Strategy()

    result = strategy.generate_signals(_make_df(close))

    warmup_slice = result.iloc[:WARMUP]
    assert (warmup_slice["signal"] == 0).all()
    assert warmup_slice["momentum_tercile_threshold"].isna().all()


def test_signals_occur_after_warmup() -> None:
    """Sanity: на достатньо довгому й варіативному ряді після розігріву реально є вхід і вихід."""
    n = 500
    close = _make_synthetic_close(n)
    strategy = Momentum30Strategy()

    result = strategy.generate_signals(_make_df(close))

    assert (result["signal"] == 1).sum() >= 1
    assert (result["signal"] == -1).sum() >= 1


def test_entry_fires_only_on_transition_not_every_active_bar() -> None:
    """
    1 має з'являтись лише на першому барі активного стану, а не на кожному
    барі, поки стан лишається активним (інакше це не "перехід", а щобарова
    ознака) — так само для -1 і неактивного стану.
    """
    n = 500
    close = _make_synthetic_close(n)
    strategy = Momentum30Strategy()

    result = strategy.generate_signals(_make_df(close))

    state = result["momentum_state_active"].to_numpy()
    signal = result["signal"].to_numpy()

    for i in range(1, len(result)):
        if state[i] and state[i - 1]:
            # Стан лишився активним -> це НЕ новий вхід.
            assert signal[i] != 1
        if (not state[i]) and (not state[i - 1]):
            # Стан лишився неактивним -> це НЕ новий вихід.
            assert signal[i] != -1


def test_missing_close_column_raises_keyerror() -> None:
    """Відсутність обов'язкової колонки 'close' має явно ламати виконання."""
    df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
    strategy = Momentum30Strategy()

    with pytest.raises(KeyError):
        strategy.generate_signals(df)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lookback": 0},
        {"lookback": -5},
        {"tercile_window": 0},
        {"tercile_window": -10},
        {"tercile_pct": 0.0},
        {"tercile_pct": 1.0},
        {"tercile_pct": 1.5},
        {"tercile_pct": -0.1},
    ],
)
def test_invalid_params_raise_value_error(kwargs: dict) -> None:
    """Некоректні параметри (lookback/tercile_window <= 0, tercile_pct поза (0,1)) мають викликати ValueError."""
    with pytest.raises(ValueError):
        Momentum30Strategy(**kwargs)


def test_does_not_mutate_input_dataframe() -> None:
    """generate_signals не повинна змінювати вхідний df на місці (повертає копію)."""
    close = _make_synthetic_close(100)
    df = _make_df(close)
    original_columns = list(df.columns)
    strategy = Momentum30Strategy()

    strategy.generate_signals(df)

    assert list(df.columns) == original_columns
    assert "signal" not in df.columns
