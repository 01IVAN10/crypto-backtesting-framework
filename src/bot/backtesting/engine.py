"""
src/bot/backtesting/engine.py

Двигун бектесту: приймає БУДЬ-ЯКУ стратегію (інтерфейс Strategy з
strategies/base.py) і симулює long-only спот-торгівлю на історичних 4h-барах
з урахуванням комісій, динамічного slippage, ATR-стопу і трейлінг-стопу.
Розмір позиції рахує risk/risk_manager.py — тут ця логіка НЕ дублюється.

ПЕРЕПРОЕКТУВАННЯ ВИХОДІВ (за результатами безумовного MFE-заміру на
exploration-наборі): фіксований take-profit (6×ATR) прибрано повністю.
Верхню межу угоди тепер формує трейлінг-стоп БЕЗ порогу активації, а не
статичний рівень. Вихід із позиції — за першою з умов: початковий стоп,
трейлінг-стоп, зворотний сигнал стратегії.

Ключові правила чесності бектесту (джерело — docs/trading_decisions.md):

1. Сигнал формується на ЗАКРИТТІ бару N (strategy.generate_signals уже
   використовує лише дані до цього бару включно, без lookahead). Вхід
   виконується за ЦІНОЮ ВІДКРИТТЯ бару N+1 — ніколи за ціною бару сигналу.

2. Той самий принцип застосовано і до виходу за зворотним сигналом: сигнал
   -1 на закритті бару M виконується за ціною відкриття бару M+1, а не
   негайно на закритті бару M. Це явно не прописано в постановці задачі
   окремо для виходів, але є прямим наслідком того самого принципу
   "рішення на закритті, виконання на наступному відкритті" — інакше вихід
   мав би доступ до інформації (закриття власного бару сигналу), якою
   в реальній торгівлі на момент рішення ще не володіли б.

3. Стоп (початковий АБО вже підтягнутий трейлінгом — це ОДИН рівень,
   position.stop_loss) — ВНУТРІШНЬОБАРОВА умова: перевіряється на low
   щойно-закритого бару, без lookahead (дані бару стають відомі рівно в
   момент його закриття). Оскільки тейк-профіту більше немає, усередині
   бару лишається РІВНО ОДНА умова виходу — колишньої неоднозначності
   "SL і TP в одному барі" як класу проблем більше не існує.

4. ДВА РІЗНІ застосування ATR, які НЕ можна плутати (явно розділені й
   закоментовані в коді нижче):
       a) entry_atr — ATR(14) з ОСТАННЬОГО закритого бару ПЕРЕД входом,
          фіксується ОДИН РАЗ при вході. Визначає початковий стоп
          (entry_price − ATR_STOP_MULT×entry_atr) І сайзинг у
          risk/risk_manager.py. Не змінюється протягом усієї угоди.
       b) трейл-дистанція = TRAIL_ATR_MULT × max(ATR ПОТОЧНОГО щойно-закритого
          бару, entry_atr) — плаває щобару вгору разом із поточною волатильністю
          (мета: ширший трейл на великих рухах), але НЕ звужується нижче
          entry_atr, коли волатильність падає. Без цієї нижньої межі
          дистанція (а не сам рівень стопу) звужувалась би симетрично й
          підіймала стоп на звичайному відкоті — захист "рухається лише
          вгору" (правило 5) цього не ловить, бо контролює РІВЕНЬ, а не
          ДИСТАНЦІЮ.

5. Трейлінг-стоп — БЕЗ порогу активації (self-taking-over): на кожному
   барі stop = max(поточний stop, running_max_high_since_entry − трейл-
   дистанція_цього_бару). На старті поточний stop == initial_stop, тож
   трейл природно "перебирає керування", щойно його кандидат перевищує
   initial_stop — без окремого прапорця чи порогу +3×ATR, як було раніше.
   Рухається ЛИШЕ вгору: якщо на якомусь барі поточний ATR різко
   розширився і формула дала НИЖЧИЙ кандидат, ніж уже піднятий стоп —
   max() з поточним рівнем захищає від відкату вниз (окремо перевірено
   тестом). Як і раніше (лаг, рule 5 попередньої версії докстрінга),
   running_max і трейл-дистанція ЦЬОГО бару впливають на перевірку SL
   лише з НАСТУПНОГО бару — інакше можна було б "заднім числом" врятувати
   угоду, що на цьому ж барі вже пробила старий (ще не піднятий) стоп.
   Причина виходу в журналі — "trailing_stop", якщо на момент виходу
   position.stop_loss > initial_stop (трейл хоч раз перебрав керування),
   інакше — "stop_loss" (спрацював ще не піднятий початковий рівень).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from bot.risk.risk_manager import calculate_position_size
from bot.strategies.base import Strategy
from config import settings

logger = logging.getLogger(__name__)

# Обов'язкові колонки вхідного df. "signal" сюди не входить — його додає
# сама стратегія всередині run() через strategy.generate_signals(df).
REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close", "atr_14"]

# 4h-бар: 6 барів на добу × 365 діб — коефіцієнт річної анулізації для Sharpe.
# Це ДЕФОЛТ конструктора BacktestEngine (нижче), а не жорстко зашитий
# коефіцієнт: для інших таймфреймів (наприклад, 1d — momentum.py,
# docs/trading_decisions.md) передавай bars_per_year=365 явно при створенні
# BacktestEngine. Існуючі виклики без цього аргументу поведінки не змінюють.
BARS_PER_YEAR_4H = 6 * 365


@dataclass
class Trade:
    """
    Один завершений трейд — рядок журналу угод.

    Поля нижче за allow_favorable/adverse-excursion блоком — чиста
    інструментація для діагностики, не впливають на жодне торгове рішення.
    Додані, щоб перевірити гіпотезу про розрив реалізованого R:R (ATR-одиниці
    проти доларових) через обрізання розміру стелею MAX_POSITION_PCT.
    """

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: float
    notional: float
    exit_reason: str  # "stop_loss" | "trailing_stop" | "reverse_signal" | "end_of_data"
    entry_commission: float
    exit_commission: float
    entry_slippage_pct: float
    exit_slippage_pct: float
    gross_pnl: float
    net_pnl: float
    return_pct: float

    # --- Інструментація (діагностика, не впливає на торгові рішення) ---
    atr_at_entry: float  # ATR(14), зафіксований при вході (визначає initial_stop і сайзинг)
    mfe_atr: float  # maximum favorable excursion за час утримання, в одиницях ATR-на-вході
    mae_atr: float  # maximum adverse excursion за час утримання, в одиницях ATR-на-вході
    trailing_activated: bool  # чи перебрав трейл керування (stop_loss > initial_stop) хоч раз
    max_favorable_price: float  # найвища ціна (high), досягнута з моменту входу
    max_favorable_time: pd.Timestamp  # час бару, на якому досягнуто max_favorable_price
    position_capped: bool  # чи обрізала стеля MAX_POSITION_PCT розрахунковий розмір при вході
    actual_risk_pct: float  # фактичний $-ризик угоди (size × stop_distance) як частка балансу входу
    bars_held: int  # кількість барів від входу до виходу (exit_idx - entry_idx)


@dataclass
class BacktestResult:
    """Повний результат прогону: журнал угод, equity curve, метрики, лічильники."""

    trades: list[Trade]
    equity_curve: pd.Series
    metrics: dict[str, float]
    skipped_min_notional_count: int

    def trades_to_dataframe(self) -> pd.DataFrame:
        """Перетворює журнал угод у DataFrame, готовий для збереження в CSV."""
        columns = [f.name for f in fields(Trade)]
        if not self.trades:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([t.__dict__ for t in self.trades], columns=columns)

    def save_trades_csv(self, path: str) -> None:
        """Зберігає журнал угод у CSV за вказаним шляхом."""
        df = self.trades_to_dataframe()
        df.to_csv(path, index=False)
        logger.info("Журнал угод збережено: %s (%d угод)", path, len(df))


@dataclass
class _OpenPosition:
    """Внутрішній стан відкритої позиції під час циклу бектесту (не публічний API)."""

    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    entry_atr: float  # ATR-на-вході (ФІКСОВАНИЙ) — визначає initial_stop і сайзинг, НЕ трейл
    position_size: float
    notional: float
    entry_commission: float
    entry_slippage_pct: float
    initial_stop: float  # НЕЗМІННИЙ рівень entry_price − ATR_STOP_MULT×entry_atr — підлога трейлу
    stop_loss: float  # ПОТОЧНИЙ активний стоп: max(initial_stop, усі попередні трейл-кандидати)
    highest_high_since_entry: float  # ЛАГОВАНИЙ трекер лише для трейлінгу (правило 5) — не чіпати
    position_capped: bool  # діагностика sizing, зафіксована один раз при вході
    actual_risk_pct: float  # діагностика sizing, зафіксована один раз при вході

    # --- Інструментація для MFE/MAE: НЕ лаговані, оновлюються на КОЖНОМУ
    # барі, поки позиція відкрита, включно з баром виходу (на відміну від
    # highest_high_since_entry вище). Не використовуються в жодному
    # торговому рішенні — лише для журналу угод.
    mfe_high_since_entry: float = 0.0
    mfe_high_time: pd.Timestamp | None = None
    mae_low_since_entry: float = 0.0


def _true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range бару: max(high-low, |high-prev_close|, |low-prev_close|).

    Рахується тут локально (а не імпортується з приватної функції
    data_pipeline/indicators.py) — потрібна лише для формули динамічного
    slippage; ATR-індикатор для стопу/трейлінгу вже прийшов готовим у
    вхідному df як колонка atr_14.
    """
    prev_close = df["close"].shift(1)
    range_hl = df["high"] - df["low"]
    range_hc = (df["high"] - prev_close).abs()
    range_lc = (df["low"] - prev_close).abs()
    return pd.concat([range_hl, range_hc, range_lc], axis=1).max(axis=1)


def _effective_slippage_pct(
    true_range: float,
    atr_prev: float,
    base_slippage_pct: float,
    cap_mult: float,
) -> float:
    """
    Динамічний slippage: BASE_SLIPPAGE × max(1, TR_бару / ATR_попереднього_бару),
    обрізаний зверху кепом cap_mult × BASE_SLIPPAGE.

    ATR_попереднього_бару тут — ЩЕ ОДНЕ, ТРЕТЄ застосування ATR у цьому модулі
    (окремо від entry_atr і трейл-дистанції, див. правило 4 докстрінга модуля):
    навмисно ЗМІЩЕНИЙ на один бар назад (не поточний, не entry) — це формула
    реалістичності ВИКОНАННЯ ордера, не пов'язана з рівнями стопу.

    Якщо ATR попереднього бару невідомий (NaN або <= 0 — можливо лише на
    самому початку серії, до прогріву data_pipeline/indicators.py) —
    використовується множник 1.0 (лише базовий рівень), а не помилка чи
    NaN, що зупинив би весь бектест.
    """
    if atr_prev is None or pd.isna(atr_prev) or atr_prev <= 0:
        mult = 1.0
    else:
        mult = max(1.0, true_range / atr_prev)
        mult = min(mult, cap_mult)
    return base_slippage_pct * mult


def compute_buy_and_hold_benchmark(df: pd.DataFrame, initial_balance: float) -> dict[str, float]:
    """
    Бенчмарк buy-and-hold: купівля на відкритті першого бару df, утримання до
    закриття останнього бару, без комісій/slippage — стандартний вигляд
    ринкового бенчмарку, з яким звіт порівнює стратегію (gate-критерій
    "Перевершує buy-and-hold BTC", docs/trading_decisions.md).

    :param df: той самий період, що й прогін стратегії (exploration-набір).
    :param initial_balance: стартовий капітал, для порівнянності з equity
        curve стратегії.
    :raises ValueError: якщо df порожній.
    """
    if df.empty:
        raise ValueError("Порожній df — неможливо порахувати buy-and-hold бенчмарк.")

    entry_price = float(df.iloc[0]["open"])
    exit_price = float(df.iloc[-1]["close"])
    total_return = exit_price / entry_price - 1.0
    final_equity = initial_balance * (1.0 + total_return)

    equity_curve = initial_balance * (df["close"].astype(float) / entry_price)
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = float(-drawdown.min())

    return {
        "total_return": total_return,
        "final_equity": final_equity,
        "max_drawdown": max_drawdown,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


class BacktestEngine:
    """
    Двигун бектесту long-only спот-торгівлі з ATR-стопом, трейлінгом без
    порогу активації, динамічним slippage і комісіями на обидві сторони угоди.
    Take-profit відсутній — верхню межу формує трейлінг-стоп.

    Приймає БУДЬ-ЯКИЙ об'єкт Strategy (інтерфейс strategies/base.py) — не
    прив'язаний до конкретної стратегії. EMACrossStrategy у scripts/backtest.py
    використана лише як перший тестовий стенд для перевірки коректності
    самого двигуна (docs/trading_decisions.md, розділ "Baseline").
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_balance: float = settings.INITIAL_BALANCE,
        commission_pct: float = settings.COMMISSION_PCT,
        base_slippage_pct: float = settings.BASE_SLIPPAGE_PCT,
        slippage_cap_mult: float = settings.SLIPPAGE_CAP_MULT,
        risk_pct: float = settings.RISK_PCT,
        max_position_pct: float = settings.MAX_POSITION_PCT,
        atr_stop_mult: float = settings.ATR_STOP_MULT,
        trail_atr_mult: float = settings.TRAIL_ATR_MULT,
        min_notional: float = settings.DEFAULT_MIN_NOTIONAL,
        bars_per_year: float = BARS_PER_YEAR_4H,
    ) -> None:
        """
        :param strategy: об'єкт зі спільним інтерфейсом Strategy.generate_signals(df).
        :param initial_balance: стартовий баланс, USDT.
        :param commission_pct: комісія біржі НА КОЖНУ СТОРОНУ угоди (у дробах).
        :param base_slippage_pct: базовий рівень slippage (у дробах) при
            "нормальній" волатильності бару виконання.
        :param slippage_cap_mult: кеп множника динамічного slippage.
        :param risk_pct, max_position_pct, atr_stop_mult: параметри, що
            передаються напряму в risk/risk_manager.calculate_position_size
            (визначають сайзинг і initial_stop).
        :param trail_atr_mult: множник ATR ПОТОЧНОГО бару для трейл-дистанції
            (окремо від atr_stop_mult, який працює лише з entry_atr — правило 4).
        :param min_notional: мінімальний нотіонал ордера біржі.
        :param bars_per_year: коефіцієнт річної анулізації Sharpe (ЛИШЕ
            довідкової метрики sharpe_ratio_informational_only — жодне
            торгове рішення від цього параметра не залежить). За
            замовчуванням BARS_PER_YEAR_4H (6 барів/добу × 365 — коректно
            для 4h-даних). Для інших таймфреймів передавай явно, напр.
            bars_per_year=365 для денних барів (Momentum30Strategy,
            scripts/backtest_momentum.py) — інакше Sharpe буде занижений/
            завищений у sqrt(6) разів.
        :raises ValueError: при некоректних параметрах витрат/капіталу.
        """
        if initial_balance <= 0:
            raise ValueError(f"initial_balance має бути додатним, отримано {initial_balance!r}.")
        if commission_pct < 0:
            raise ValueError(f"commission_pct не може бути від'ємним, отримано {commission_pct!r}.")
        if base_slippage_pct < 0:
            raise ValueError(
                f"base_slippage_pct не може бути від'ємним, отримано {base_slippage_pct!r}."
            )
        if slippage_cap_mult < 1.0:
            raise ValueError(f"slippage_cap_mult має бути >= 1.0, отримано {slippage_cap_mult!r}.")
        if trail_atr_mult <= 0:
            raise ValueError(f"trail_atr_mult має бути додатним, отримано {trail_atr_mult!r}.")
        if bars_per_year <= 0:
            raise ValueError(f"bars_per_year має бути додатним, отримано {bars_per_year!r}.")

        self.strategy = strategy
        self.initial_balance = initial_balance
        self.commission_pct = commission_pct
        self.base_slippage_pct = base_slippage_pct
        self.slippage_cap_mult = slippage_cap_mult
        self.risk_pct = risk_pct
        self.max_position_pct = max_position_pct
        self.atr_stop_mult = atr_stop_mult
        self.trail_atr_mult = trail_atr_mult
        self.min_notional = min_notional
        self.bars_per_year = bars_per_year

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Запускає повний прогін бектесту на df.

        :param df: OHLCV + індикатори (щонайменше REQUIRED_COLUMNS),
            відсортований за часом за зростанням. Колонка "signal" НЕ
            потрібна у вхідних даних — вона генерується всередині виклику
            self.strategy.generate_signals(df).
        :return: BacktestResult з журналом угод, equity curve і метриками.
        :raises KeyError: якщо в df відсутня хоч одна з обов'язкових колонок.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"У df відсутні обов'язкові колонки для BacktestEngine: {missing}")

        df = self.strategy.generate_signals(df).reset_index(drop=True)
        if "signal" not in df.columns:
            raise KeyError(
                "strategy.generate_signals(df) не додав колонку 'signal' — "
                "порушення контракту інтерфейсу Strategy (strategies/base.py)."
            )

        true_range = _true_range(df)
        atr_prev = df["atr_14"].shift(1)

        n = len(df)
        balance = self.initial_balance
        position: _OpenPosition | None = None
        pending_reverse_exit = False
        trades: list[Trade] = []
        equity_curve = np.empty(n, dtype=float)
        skipped_min_notional_count = 0

        for j in range(n):
            row = df.iloc[j]

            # --- 1. Відкладені події, заплановані на ВІДКРИТТЯ цього бару ---
            # (зворотний сигнал, згенерований на закритті ПОПЕРЕДНЬОГО бару —
            # правило 2 з докстрінга модуля: без lookahead на закриття власного бару)
            if position is not None and pending_reverse_exit:
                trade = self._close_position(
                    position=position,
                    exit_idx=j,
                    exit_price_raw=float(row["open"]),
                    exit_time=row["datetime"],
                    exit_reason="reverse_signal",
                    true_range_bar=float(true_range.iloc[j]),
                    atr_prev_bar=atr_prev.iloc[j],
                )
                balance += trade.position_size * trade.exit_price - trade.exit_commission
                trades.append(trade)
                position = None
                pending_reverse_exit = False

            # --- 2. Вхід на ВІДКРИТТІ цього бару за сигналом, згенерованим
            #        на закритті ПОПЕРЕДНЬОГО бару (правило 1) ---
            if position is None and j > 0 and int(df.loc[j - 1, "signal"]) == 1:
                new_position, skipped = self._try_open_position(
                    balance=balance,
                    entry_idx=j,
                    entry_time=row["datetime"],
                    open_price_raw=float(row["open"]),
                    entry_atr=df.loc[j - 1, "atr_14"],
                    true_range_bar=float(true_range.iloc[j]),
                    atr_prev_bar=atr_prev.iloc[j],
                )
                if skipped:
                    skipped_min_notional_count += 1
                elif new_position is not None:
                    balance -= new_position.notional + new_position.entry_commission
                    position = new_position

            # --- 3. Внутрішньобарова перевірка стопу щойно-закритого бару
            #        (правило 3), потім трейлінг і зворотний сигнал ---
            if position is not None:
                # Інструментація MFE/MAE: оновлюється БЕЗУМОВНО на кожному
                # барі, поки позиція відкрита — включно з цим самим баром,
                # навіть якщо він виявиться баром виходу через стоп нижче.
                # На відміну від highest_high_since_entry (трейлінг, лаговано,
                # правило 5), тут лаг НЕ потрібен: це чиста діагностика
                # "що показав ринок за час утримання", вона нічого не вирішує.
                bar_high = float(row["high"])
                bar_low = float(row["low"])
                if bar_high > position.mfe_high_since_entry:
                    position.mfe_high_since_entry = bar_high
                    position.mfe_high_time = row["datetime"]
                position.mae_low_since_entry = min(position.mae_low_since_entry, bar_low)

                hit_sl = bool(bar_low <= position.stop_loss)

                if hit_sl:
                    # Причина виходу залежить від того, чи трейл уже перебрав
                    # керування (position.stop_loss > initial_stop) — не від
                    # окремого прапорця (правило 5: "self-taking-over").
                    exit_reason = (
                        "trailing_stop" if position.stop_loss > position.initial_stop
                        else "stop_loss"
                    )
                    trade = self._close_position(
                        position=position, exit_idx=j, exit_price_raw=position.stop_loss,
                        exit_time=row["datetime"], exit_reason=exit_reason,
                        true_range_bar=float(true_range.iloc[j]), atr_prev_bar=atr_prev.iloc[j],
                    )
                    balance += trade.position_size * trade.exit_price - trade.exit_commission
                    trades.append(trade)
                    position = None
                else:
                    # Зворотний сигнал на закритті ЦЬОГО бару -> вихід запланований
                    # на відкриття НАСТУПНОГО (крок 1 наступної ітерації).
                    if int(row["signal"]) == -1:
                        pending_reverse_exit = True

                    # Правило 5: new high цього бару оновлює стан позиції, але
                    # піднятий стоп почне впливати на перевірку лише з НАСТУПНОГО
                    # бару (наступна ітерація циклу читає вже оновлений position.stop_loss).
                    position.highest_high_since_entry = max(
                        position.highest_high_since_entry, bar_high
                    )

                    # Правило 4b: трейл-дистанція рахується від ATR ПОТОЧНОГО
                    # (щойно-закритого) бару — НЕ від entry_atr. Плаває щобару.
                    #
                    # ТОЧКОВА ПРАВКА: дистанція = TRAIL_ATR_MULT × max(поточний
                    # ATR, entry_atr), а не просто поточний ATR. Причина:
                    # поточний ATR розширює трейл у великих рухах (це мета), але
                    # СИМЕТРИЧНО звужує його при ПАДІННІ волатильності — рівень
                    # стопу тоді підстрибує вгору й вибиває позицію на звичайному
                    # відкоті. Захист "рухається лише вгору" нижче цього не
                    # ловить: він контролює РІВЕНЬ стопу, а не ДИСТАНЦІЮ до нього,
                    # тож звужена дистанція все одно могла підняти рівень.
                    # entry_atr як нижня межа дистанції — та сама фіксована
                    # величина, що вже визначає initial_stop і сайзинг.
                    current_bar_atr = row["atr_14"]
                    if current_bar_atr is not None and not pd.isna(current_bar_atr) and (
                        current_bar_atr > 0
                    ):
                        effective_atr_for_trail = max(float(current_bar_atr), position.entry_atr)
                        trail_candidate = (
                            position.highest_high_since_entry
                            - self.trail_atr_mult * effective_atr_for_trail
                        )
                        # Правило 5: рухається ЛИШЕ вгору. Якщо поточний ATR
                        # різко розширився і candidate вийшов НИЖЧИМ за вже
                        # піднятий стоп — max() захищає від відкату вниз.
                        if trail_candidate > position.stop_loss:
                            position.stop_loss = trail_candidate

            equity_curve[j] = balance + (
                position.position_size * float(row["close"]) if position is not None else 0.0
            )

        # Примусове закриття позиції, що лишилась відкритою на кінець серії —
        # інакше вона випала б із журналу угод і метрик (equity curve
        # приховала б нереалізований P&L замість явного трейду в звіті).
        if position is not None:
            last_row = df.iloc[n - 1]
            trade = self._close_position(
                position=position, exit_idx=n - 1, exit_price_raw=float(last_row["close"]),
                exit_time=last_row["datetime"], exit_reason="end_of_data",
                true_range_bar=float(true_range.iloc[n - 1]), atr_prev_bar=atr_prev.iloc[n - 1],
            )
            balance += trade.position_size * trade.exit_price - trade.exit_commission
            trades.append(trade)
            equity_curve[n - 1] = balance
            position = None

        equity_series = pd.Series(equity_curve, index=pd.to_datetime(df["datetime"]))
        metrics = self._compute_metrics(trades=trades, equity_curve=equity_series)

        logger.info(
            "Бектест завершено (%s): %d угод, %d пропущено (below_min_notional). "
            "total_return=%.2f%%",
            self.strategy.name, len(trades), skipped_min_notional_count,
            metrics["total_return"] * 100,
        )

        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            metrics=metrics,
            skipped_min_notional_count=skipped_min_notional_count,
        )

    def _try_open_position(
        self,
        balance: float,
        entry_idx: int,
        entry_time: pd.Timestamp,
        open_price_raw: float,
        entry_atr: float,
        true_range_bar: float,
        atr_prev_bar: float,
    ) -> tuple[_OpenPosition | None, bool]:
        """
        Намагається відкрити позицію на вхідному сигналі.

        entry_atr фіксується тут ОДИН РАЗ (значення з ОСТАННЬОГО закритого
        бару перед входом, без lookahead) і зберігається в _OpenPosition на
        всю тривалість угоди — визначає initial_stop і НЕ перераховується
        (правило 4a). Розмір позиції рахує calculate_position_size() з
        risk/risk_manager.py — тут ця логіка НЕ дублюється.

        :return: (позиція або None, чи пропущено через below_min_notional).
            Якщо ATR попереднього бару ще NaN (warm-up-період
            data_pipeline/indicators.py) — сигнал ігнорується (це не skip
            від risk_manager, а відсутність вхідних даних для розрахунку),
            повертається (None, False).
        """
        if entry_atr is None or pd.isna(entry_atr) or entry_atr <= 0:
            logger.warning(
                "Сигнал входу на барі %d проігноровано: ATR(14) попереднього бару "
                "невідомий (warm-up-період data_pipeline/indicators.py).",
                entry_idx,
            )
            return None, False

        slippage_pct = _effective_slippage_pct(
            true_range=true_range_bar,
            atr_prev=atr_prev_bar,
            base_slippage_pct=self.base_slippage_pct,
            cap_mult=self.slippage_cap_mult,
        )
        # Slippage погіршує ціну виконання: для купівлі це вища ціна.
        entry_price = open_price_raw * (1.0 + slippage_pct)

        sizing = calculate_position_size(
            balance=balance,
            entry_price=entry_price,
            atr=float(entry_atr),
            has_open_position=False,  # engine викликає це лише коли позиції немає
            min_notional=self.min_notional,
            risk_pct=self.risk_pct,
            max_position_pct=self.max_position_pct,
            atr_stop_mult=self.atr_stop_mult,
        )

        if not sizing.approved:
            logger.info(
                "Вхід на барі %d пропущено (%s): balance=%.2f, entry_price=%.2f, atr=%.4f",
                entry_idx, sizing.reason, balance, entry_price, entry_atr,
            )
            return None, True

        entry_commission = sizing.notional * self.commission_pct

        # --- Діагностика sizing (чиста інструментація, НЕ впливає на sizing.position_size
        # чи будь-яке торгове рішення — той сам розмір/стоп беруться напряму з sizing) ---
        # raw_size/raw_notional повторно використовують sizing.risk_amount і
        # sizing.stop_distance — ВЖЕ пораховані й повернуті calculate_position_size()
        # (крок 3 його docstring'а), тут лише застосовано ту саму формулу до
        # цих готових чисел, щоб визначити, чи спрацювала стеля (крок 4), а не
        # заново вирішено, яким має бути розмір.
        raw_size = sizing.risk_amount / sizing.stop_distance
        raw_notional = raw_size * entry_price
        position_capped = bool(raw_notional > sizing.notional)
        actual_risk_amount = sizing.position_size * sizing.stop_distance
        actual_risk_pct = actual_risk_amount / balance

        position = _OpenPosition(
            entry_idx=entry_idx,
            entry_time=entry_time,
            entry_price=entry_price,
            entry_atr=float(entry_atr),
            position_size=sizing.position_size,
            notional=sizing.notional,
            entry_commission=entry_commission,
            entry_slippage_pct=slippage_pct,
            initial_stop=sizing.stop_loss_price,
            stop_loss=sizing.stop_loss_price,
            highest_high_since_entry=entry_price,
            position_capped=position_capped,
            actual_risk_pct=actual_risk_pct,
            mfe_high_since_entry=entry_price,
            mfe_high_time=entry_time,
            mae_low_since_entry=entry_price,
        )
        logger.info(
            "Відкрито позицію на барі %d: entry=%.2f, розмір=%.8f, initial_stop=%.2f, "
            "capped=%s, actual_risk_pct=%.4f",
            entry_idx, entry_price, position.position_size, position.initial_stop,
            position_capped, actual_risk_pct,
        )
        return position, False

    def _close_position(
        self,
        position: _OpenPosition,
        exit_idx: int,
        exit_price_raw: float,
        exit_time: pd.Timestamp,
        exit_reason: str,
        true_range_bar: float,
        atr_prev_bar: float,
    ) -> Trade:
        """
        Формує Trade для закриття позиції, застосовуючи slippage (що
        ПОГІРШУЄ ціну продажу, тобто занижує її) і комісію на вихід.

        :param exit_price_raw: "теоретична" ціна виконання ДО slippage —
            рівень стопу (початкового або трейлінгового) або ціна
            відкриття/закриття бару (для reverse_signal / end_of_data відповідно).
        """
        slippage_pct = _effective_slippage_pct(
            true_range=true_range_bar,
            atr_prev=atr_prev_bar,
            base_slippage_pct=self.base_slippage_pct,
            cap_mult=self.slippage_cap_mult,
        )
        # Slippage погіршує ціну виконання: для продажу це нижча ціна.
        exit_price = exit_price_raw * (1.0 - slippage_pct)

        proceeds = position.position_size * exit_price
        exit_commission = proceeds * self.commission_pct

        gross_pnl = proceeds - position.notional
        total_cost = position.notional + position.entry_commission
        net_pnl = (proceeds - exit_commission) - total_cost
        return_pct = net_pnl / total_cost

        # --- Інструментація: MFE/MAE в одиницях ATR-на-вході, bars_held.
        # mfe_high_since_entry ratchet-иться лише вгору від entry_price,
        # mae_low_since_entry — лише вниз, тож обидві різниці невід'ємні
        # за побудовою (без додаткового clip).
        mfe_atr = (position.mfe_high_since_entry - position.entry_price) / position.entry_atr
        mae_atr = (position.entry_price - position.mae_low_since_entry) / position.entry_atr
        bars_held = exit_idx - position.entry_idx
        trailing_activated = bool(position.stop_loss > position.initial_stop)

        logger.info(
            "Закрито позицію (%s): вхід %.2f -> вихід %.2f, розмір %.8f, "
            "net_pnl=%.4f USDT (%.2f%%)",
            exit_reason, position.entry_price, exit_price, position.position_size,
            net_pnl, return_pct * 100,
        )

        return Trade(
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            position_size=position.position_size,
            notional=position.notional,
            exit_reason=exit_reason,
            entry_commission=position.entry_commission,
            exit_commission=exit_commission,
            entry_slippage_pct=position.entry_slippage_pct,
            exit_slippage_pct=slippage_pct,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_pct=return_pct,
            atr_at_entry=position.entry_atr,
            mfe_atr=mfe_atr,
            mae_atr=mae_atr,
            trailing_activated=trailing_activated,
            max_favorable_price=position.mfe_high_since_entry,
            max_favorable_time=position.mfe_high_time,
            position_capped=position.position_capped,
            actual_risk_pct=position.actual_risk_pct,
            bars_held=bars_held,
        )

    def _compute_metrics(self, trades: list[Trade], equity_curve: pd.Series) -> dict[str, float]:
        """
        Рахує підсумкові метрики бектесту.

        Sharpe тут — ДОВІДКОВИЙ показник, НЕ gate-критерій (docs/trading_decisions.md):
        розподіл дохідностей 4h-барів має важкі хвости (excess kurtosis ≈ 8 за EDA),
        Sharpe припускає нормальність розподілу і буде оптимістичним.
        """
        final_equity = float(equity_curve.iloc[-1]) if len(equity_curve) else self.initial_balance
        total_return = final_equity / self.initial_balance - 1.0

        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = float(-drawdown.min()) if len(drawdown) else 0.0

        n_trades = len(trades)
        winning = [t for t in trades if t.net_pnl > 0]
        losing = [t for t in trades if t.net_pnl <= 0]
        win_rate = len(winning) / n_trades if n_trades > 0 else 0.0

        gross_profit = sum(t.net_pnl for t in winning)
        gross_loss = abs(sum(t.net_pnl for t in losing))
        if n_trades == 0:
            profit_factor = 0.0
        elif gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf")  # жодної збиткової угоди

        avg_win = (gross_profit / len(winning)) if winning else 0.0
        avg_loss = (-gross_loss / len(losing)) if losing else 0.0

        bar_returns = equity_curve.pct_change().dropna()
        if len(bar_returns) > 1 and bar_returns.std(ddof=0) > 0:
            sharpe = float(
                bar_returns.mean() / bar_returns.std(ddof=0) * np.sqrt(self.bars_per_year)
            )
        else:
            sharpe = 0.0

        return {
            "total_return": total_return,
            "final_equity": final_equity,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "trades_count": float(n_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "sharpe_ratio_informational_only": sharpe,
        }
