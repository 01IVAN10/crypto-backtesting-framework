"""
tests/test_backtest_engine.py

Тести для src/bot/backtesting/engine.py.

Двигун генерично приймає будь-яку Strategy — тут використана мінімальна
тестова _FixedSignalStrategy (наперед задана колонка 'signal'), щоб
ізолювати перевірку самого двигуна від логіки конкретної EMACrossStrategy.

Ключові сценарії (за завданням — обов'язкові):
- відсутність lookahead: вхід НЕ за ціною/часом бару сигналу, а за
  відкриттям НАСТУПНОГО бару;
- коректність обліку комісій: 0.1% нараховується окремо на нотіонал входу
  і окремо на виручку виходу, і саме ця сума зменшує net_pnl.

Додатково (посилюють впевненість у "чесності" двигуна):
- той самий принцип "без lookahead" застосований і до виходу за зворотним
  сигналом стратегії;
- угода, що не проходить min_notional, пропускається і рахується окремим
  лічильником, а не округлюється вгору чи ігнорується мовчки.

Тести на перепроектовані виходи (take-profit прибрано, трейлінг без порогу
активації, trail-дистанція від ATR ПОТОЧНОГО бару — не entry_atr):
- трейл перебирає керування, щойно кандидат перевищує initial_stop, БЕЗ
  окремого порогу активації (раніше — +3×ATR);
- trail-дистанція використовує ATR саме ПОТОЧНОГО бару, а не зафіксований
  entry_atr — перевірено сценарієм, де ці два розрахунки дають РІЗНІ рівні;
- trail рухається лише вгору навіть якщо ATR різко розширився і формула
  того бару дала б нижчий рівень.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bot.backtesting.engine import BacktestEngine
from bot.strategies.base import Strategy


class _FixedSignalStrategy(Strategy):
    """
    Мінімальна тестова стратегія: повертає наперед задану колонку 'signal',
    ігноруючи фактичний вміст df. Потрібна лише для ізольованого тестування
    BacktestEngine окремо від логіки конкретних стратегій.
    """

    def __init__(self, signals: list[int]) -> None:
        super().__init__(name="fixed_signal_test_strategy")
        self._signals = signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = self._signals
        return df


def _make_flat_df(n: int, base_price: float = 1000.0, atr: float = 10.0) -> pd.DataFrame:
    """
    Будує "спокійний" df на n барів: ціна не рухається настільки, щоб самі
    собою (без явних override) спрацювали SL (2×ATR=20) чи TP (6×ATR=60)
    від будь-якої ціни входу, близької до base_price.
    """
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [base_price] * n,
            "high": [base_price + 2.0] * n,
            "low": [base_price - 2.0] * n,
            "close": [base_price] * n,
            "atr_14": [atr] * n,
        }
    )


def test_entry_executes_at_next_bar_open_not_signal_bar() -> None:
    """
    Сигнал 1 на закритті бару 1 -> вхід МАЄ виконатись за ціною відкриття
    бару 2, а не за відкриттям/закриттям бару 1 (перевірка lookahead).
    """
    df = _make_flat_df(n=6)
    df.loc[1, ["open", "close"]] = [1000.0, 1005.0]  # ціни бару сигналу — НЕ мають потрапити у вхід
    df.loc[2, ["open", "high", "low", "close"]] = [1050.0, 1052.0, 1048.0, 1050.0]

    strategy = _FixedSignalStrategy(signals=[0, 1, 0, 0, 0, 0])
    engine = BacktestEngine(
        strategy=strategy,
        initial_balance=1000.0,
        commission_pct=0.0,
        base_slippage_pct=0.0,  # ізолюємо перевірку ціни від slippage
        min_notional=0.0,
    )

    result = engine.run(df)

    assert len(result.trades) >= 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(1050.0)
    assert trade.entry_time == df.loc[2, "datetime"]
    # Явно переконуємось, що це НЕ ціни бару сигналу (бар 1).
    assert trade.entry_price != pytest.approx(1000.0)
    assert trade.entry_price != pytest.approx(1005.0)


def test_reverse_signal_exit_executes_at_next_bar_open_not_signal_bar() -> None:
    """
    Той самий принцип "без lookahead", застосований до виходу: зворотний
    сигнал (-1) на закритті бару M виконується за відкриттям бару M+1,
    а не за ціною бару M.
    """
    df = _make_flat_df(n=6)
    df.loc[3, ["close"]] = [995.0]  # ціна бару сигналу виходу — НЕ має потрапити у вихід
    df.loc[4, ["open", "high", "low", "close"]] = [1010.0, 1012.0, 1008.0, 1010.0]

    strategy = _FixedSignalStrategy(signals=[0, 1, 0, -1, 0, 0])
    engine = BacktestEngine(
        strategy=strategy,
        initial_balance=1000.0,
        commission_pct=0.0,
        base_slippage_pct=0.0,
        min_notional=0.0,
    )

    result = engine.run(df)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "reverse_signal"
    assert trade.exit_price == pytest.approx(1010.0)
    assert trade.exit_time == df.loc[4, "datetime"]
    assert trade.exit_price != pytest.approx(995.0)
    assert trade.exit_price != pytest.approx(1000.0)  # bar3 open теж не мав потрапити у вихід


def test_commission_charged_separately_on_entry_and_exit() -> None:
    """
    Комісія 0.1% має нараховуватись ОКРЕМО на нотіонал входу і ОКРЕМО на
    виручку виходу, і саме ця сума має пояснювати різницю gross_pnl -> net_pnl.
    Порівнюємо прогін з комісією 0 і з комісією 0.1% на ідентичному русі ціни.
    """
    df = _make_flat_df(n=4)
    df.loc[2, ["open", "high", "low", "close"]] = [1050.0, 1052.0, 1048.0, 1050.0]
    df.loc[3, ["open", "high", "low", "close"]] = [1050.0, 1060.0, 1040.0, 1055.0]

    strategy_kwargs = dict(min_notional=0.0, base_slippage_pct=0.0, initial_balance=1000.0)

    engine_no_fee = BacktestEngine(
        strategy=_FixedSignalStrategy(signals=[0, 1, 0, 0]), commission_pct=0.0, **strategy_kwargs
    )
    engine_with_fee = BacktestEngine(
        strategy=_FixedSignalStrategy(signals=[0, 1, 0, 0]), commission_pct=0.001, **strategy_kwargs
    )

    trade_no_fee = engine_no_fee.run(df).trades[0]
    trade_with_fee = engine_with_fee.run(df).trades[0]

    # Той самий рух ціни -> той самий gross_pnl незалежно від комісії.
    assert trade_with_fee.gross_pnl == pytest.approx(trade_no_fee.gross_pnl)
    # Без комісії net_pnl == gross_pnl.
    assert trade_no_fee.net_pnl == pytest.approx(trade_no_fee.gross_pnl)

    # Комісія входу — рівно 0.1% від нотіоналу входу.
    assert trade_with_fee.entry_commission == pytest.approx(trade_with_fee.notional * 0.001)
    # Комісія виходу — рівно 0.1% від фактичної виручки (розмір × ціна виходу).
    expected_exit_commission = trade_with_fee.position_size * trade_with_fee.exit_price * 0.001
    assert trade_with_fee.exit_commission == pytest.approx(expected_exit_commission)

    # net_pnl з комісією == gross_pnl мінус обидві комісії (і жодним іншим чином).
    expected_net_pnl = (
        trade_with_fee.gross_pnl - trade_with_fee.entry_commission - trade_with_fee.exit_commission
    )
    assert trade_with_fee.net_pnl == pytest.approx(expected_net_pnl)
    # Комісія завжди зменшує результат: з комісією net_pnl менший.
    assert trade_with_fee.net_pnl < trade_no_fee.net_pnl


def test_trailing_takes_over_without_activation_threshold() -> None:
    """
    Пункт 2: трейл БЕЗ порогу активації. Нереалізований прибуток +15 (менше
    за старий поріг +3×ATR=+30) все одно МАЄ підняти стоп вище initial_stop,
    щойно кандидат його перевищує — "self-taking-over", без окремого прапорця.
    """
    df = _make_flat_df(n=3)
    # entry_atr=10 (default, бар 1) -> initial_stop = entry(1000) − 2×10 = 980.
    # trail_atr_mult=1.0 -> після бару 2 (high=1015, atr=10):
    #   running_max=1015, candidate = 1015 − 1×10 = 1005 > 980 -> стоп підтягнуто
    #   до 1005 ОДРАЗУ, хоча +15 прибутку набагато менше старого порогу +30.
    df.loc[2, ["open", "high", "low", "close"]] = [1000.0, 1015.0, 995.0, 1000.0]

    strategy = _FixedSignalStrategy(signals=[0, 1, 0])
    engine = BacktestEngine(
        strategy=strategy,
        initial_balance=1000.0,
        commission_pct=0.0,
        base_slippage_pct=0.0,
        min_notional=0.0,
        trail_atr_mult=1.0,
    )

    result = engine.run(df)

    # Позиція лишається відкритою (жоден бар не пробив стоп) -> примусово
    # закрита в кінці серії. Достатньо перевірити, що трейл уже перебрав
    # керування: max_favorable/trailing_activated відображають рівень 1005.
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.trailing_activated is True
    assert trade.max_favorable_price == pytest.approx(1015.0)


def test_trail_distance_uses_current_bar_atr_not_entry_atr() -> None:
    """
    Пункт 3: trail-дистанція = TRAIL_ATR_MULT × ATR ПОТОЧНОГО бару, НЕ
    entry_atr. entry_atr=10 (фіксований), але бар 3 приносить ATR=30.

    Якби двигун помилково використав entry_atr (10) для трейлінгу, стоп
    після бару 3 був би 1100 − 1×10 = 1090, і бар 4 (low=1075) закрив би
    угоду по trailing_stop=1090 ДО кінця серії. З ПРАВИЛЬНИМ поточним ATR
    (30) стоп = 1100 − 1×30 = 1070, low=1075 його не пробиває -> позиція
    доживає до кінця серії й закривається як end_of_data.
    """
    df = _make_flat_df(n=5)
    # entry_atr=10 (default, бар 1) -> initial_stop = 1000 − 20 = 980.
    df.loc[2, ["open", "high", "low", "close", "atr_14"]] = [1000.0, 1000.0, 998.0, 1000.0, 10.0]
    df.loc[3, ["open", "high", "low", "close", "atr_14"]] = [1000.0, 1100.0, 999.0, 1100.0, 30.0]
    df.loc[4, ["open", "high", "low", "close", "atr_14"]] = [1075.0, 1085.0, 1075.0, 1080.0, 30.0]

    strategy = _FixedSignalStrategy(signals=[0, 1, 0, 0, 0])
    engine = BacktestEngine(
        strategy=strategy,
        initial_balance=1000.0,
        commission_pct=0.0,
        base_slippage_pct=0.0,
        min_notional=0.0,
        trail_atr_mult=1.0,
    )

    result = engine.run(df)

    assert len(result.trades) == 1
    trade = result.trades[0]
    # Якби рахувалось від entry_atr, вийшло б exit_reason="trailing_stop",
    # exit_price≈1090, ще на барі 3. Коректна поведінка — доживання до кінця.
    assert trade.exit_reason == "end_of_data"
    assert trade.exit_price == pytest.approx(1080.0)


def test_trail_never_moves_down_even_if_current_atr_spikes() -> None:
    """
    Пункт 4: трейл рухається ЛИШЕ вгору. Різкий стрибок ПОТОЧНОГО ATR
    (10 -> 500) дає формулу з набагато НИЖЧИМ кандидатом за вже піднятий
    стоп — двигун МАЄ проігнорувати цей нижчий рівень (max() з поточним
    стопом), а не відкотити стоп назад униз.
    """
    df = _make_flat_df(n=5)
    # entry_atr=10 (default, бар 1) -> initial_stop = 1000 − 20 = 980.
    # Бар 2: running_max=1050, atr=10 -> candidate = 1050 − 1×10 = 1040 (S1).
    df.loc[2, ["open", "high", "low", "close", "atr_14"]] = [1000.0, 1050.0, 998.0, 1030.0, 10.0]
    # Бар 3: ATR-СПЛЕСК (atr=500), але low(1042) вище S1(1040) -> без виходу.
    # Якби max()-захисту не було: candidate = 1050 − 1×500 = 550 (набагато нижче S1).
    df.loc[3, ["open", "high", "low", "close", "atr_14"]] = [1045.0, 1048.0, 1042.0, 1045.0, 500.0]
    # Бар 4: low=800 -> вище "хибного" 550, але НИЖЧЕ справжнього S1=1040 ->
    # має спрацювати вихід рівно по S1=1040, а не залишитись відкритою.
    df.loc[4, ["open", "high", "low", "close", "atr_14"]] = [1035.0, 1038.0, 800.0, 810.0, 500.0]

    strategy = _FixedSignalStrategy(signals=[0, 1, 0, 0, 0])
    engine = BacktestEngine(
        strategy=strategy,
        initial_balance=1000.0,
        commission_pct=0.0,
        base_slippage_pct=0.0,
        min_notional=0.0,
        trail_atr_mult=1.0,
    )

    result = engine.run(df)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "trailing_stop"
    assert trade.exit_price == pytest.approx(1040.0)


def test_trail_distance_does_not_narrow_below_entry_atr() -> None:
    """
    Точкова правка: trail-дистанція = TRAIL_ATR_MULT × max(поточний ATR,
    entry_atr) — НЕ звужується нижче entry_atr, коли поточна волатильність
    падає. Без цієї нижньої межі падіння ATR (10 -> 2) симетрично звузило б
    дистанцію й підняло б РІВЕНЬ стопу без жодного нового high — "фантомний"
    stop-out на звичайному коливанні, а не на реальній слабкості ціни.

    entry_atr=10 (бар 1) -> initial_stop = 980. Бар 2: running_max=1050,
    atr=10 -> стоп підтягнуто до S1=1050-1×10=1040. Бар 3: ATR падає до 2,
    ціна не оновлює running_max (high=1038<1050) -> якби дистанція звузилась
    до 1×2=2, кандидат = 1050-1×2=1048 > 1040 підняв би стоп до 1048. Бар 4:
    low=1042 — вище S1=1040 (без виходу), але НИЖЧЕ хибного 1048 (виход би
    відбувся). З нижньою межею entry_atr дистанція лишається 1×10=10, стоп
    лишається 1040, і позиція доживає до кінця серії.
    """
    df = _make_flat_df(n=5)
    df.loc[2, ["open", "high", "low", "close", "atr_14"]] = [1000.0, 1050.0, 998.0, 1030.0, 10.0]
    df.loc[3, ["open", "high", "low", "close", "atr_14"]] = [1035.0, 1038.0, 1042.0, 1036.0, 2.0]
    df.loc[4, ["open", "high", "low", "close", "atr_14"]] = [1041.0, 1045.0, 1042.0, 1043.0, 2.0]

    strategy = _FixedSignalStrategy(signals=[0, 1, 0, 0, 0])
    engine = BacktestEngine(
        strategy=strategy,
        initial_balance=1000.0,
        commission_pct=0.0,
        base_slippage_pct=0.0,
        min_notional=0.0,
        trail_atr_mult=1.0,
    )

    result = engine.run(df)

    assert len(result.trades) == 1
    trade = result.trades[0]
    # Якби дистанція звузилась до поточного ATR=2, стоп на барі 3 стрибнув би
    # до 1048, і бар 4 (low=1042) закрив би угоду по trailing_stop=1048 ще
    # ДО кінця серії. З нижньою межею entry_atr позиція доживає до кінця.
    assert trade.exit_reason == "end_of_data"
    assert trade.exit_price == pytest.approx(1043.0)


def test_trade_skipped_and_counted_when_below_min_notional() -> None:
    """
    Якщо calculate_position_size() повертає skip (below_min_notional) —
    угода НЕ відкривається і рахується в skipped_min_notional_count, а не
    просто мовчки зникає чи округлюється вгору.
    """
    df = _make_flat_df(n=3)
    strategy = _FixedSignalStrategy(signals=[0, 1, 0])
    engine = BacktestEngine(
        strategy=strategy,
        initial_balance=1000.0,
        commission_pct=0.0,
        base_slippage_pct=0.0,
        min_notional=1_000_000_000.0,  # свідомо недосяжний мінімум
    )

    result = engine.run(df)

    assert len(result.trades) == 0
    assert result.skipped_min_notional_count == 1
