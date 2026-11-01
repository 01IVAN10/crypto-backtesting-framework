"""
src/bot/strategies/momentum.py

Momentum-30 — перша НЕ-тестова стратегія проекту (на відміну від
strategies/ema_cross.py, який лишається лише стендом для перевірки двигуна
бектесту). Кандидат пройшов вимір придатності на exploration-наборі
(крок 7.9.3 роадмапу: 4 значущі зсуви розподілу з 18), працює на денному
таймфреймі (docs/trading_decisions.md).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bot.strategies.base import Strategy
from config import settings

logger = logging.getLogger(__name__)


class Momentum30Strategy(Strategy):
    """
    Momentum-стратегія на 30-денній log-дохідності з ковзним терциль-порогом.

    Ознака (feature): log-дохідність close-to-close за lookback попередніх
    барів: momentum_return[t] = ln(close[t] / close[t - lookback]).

    Стан: "активний", коли momentum_return на цьому барі вище порогу
    верхнього терциля, порахованого на КОВЗНОМУ вікні з tercile_window
    ПОПЕРЕДНІХ значень momentum_return — бари [t - tercile_window, t - 1],
    БЕЗ самого бару t. Поріг для бару t не залежить від значення
    momentum_return[t]: ця точка порівнюється з референсним розподілом,
    який був би відомий ДО того, як вона сама з'явилась, а не бере участь
    у формуванні власного порогу. Реалізовано через .shift(1) ПЕРЕД
    .rolling() (не одне і те саме, що просто .rolling() без shift — див.
    generate_signals): .rolling() сам собою дивиться лише назад ВКЛЮЧНО з
    поточним індексом, тобто без додаткового .shift(1) вікно для бару t
    було б [t - tercile_window + 1, t] — це не lookahead у класичному сенсі
    (жодних барів ПІСЛЯ t), але порушує вимогу "поріг без t": значення
    брало б участь у визначенні власного порогу.

    Сигнали:
        1 (вхід у лонг): стан переходить з неактивного в активний.
        -1 (зворотний сигнал / вихід): стан переходить з активного в
            неактивний.
        0: решта барів (стан не змінився).

    Тільки лонг, без шортів. Фільтр EMA-200 НЕ застосовується (за
    результатами виміру придатності: чотири незалежні заміри показали, що
    він нічого не додає, а перетин momentum_30 з EMA-200-фільтром і так
    86% — docs/trading_decisions.md).

    Сигнал формується на закритті бару; вхід/вихід за ціною наступного
    бару — це відповідальність src/bot/backtesting/engine.py (той самий
    механізм "рішення на закритті, виконання на наступному відкритті", що
    й для ema_cross.py), тут лише генерується колонка 'signal'.

    Очікує в df готову колонку 'close'. ATR/EMA/інші індикатори тут не
    потрібні — вони не використовуються цією стратегією (тренд-фільтр
    свідомо відсутній).

    Крім 'signal' повертає діагностичні колонки momentum_return,
    momentum_tercile_threshold, momentum_state_active — для звітності й
    тестів на lookahead; двигун бектесту їх ігнорує (перевіряє лише
    наявність 'signal').
    """

    def __init__(
        self,
        lookback: int = settings.MOMENTUM_LOOKBACK,
        tercile_window: int = settings.MOMENTUM_TERCILE_WINDOW,
        tercile_pct: float = settings.TERCILE_PCT,
        name: str = "momentum_30",
    ) -> None:
        """
        Ініціалізує стратегію.

        Параметри:
            lookback: кількість барів для розрахунку log-дохідності
                (за замовчуванням config.settings.MOMENTUM_LOOKBACK).
            tercile_window: розмір ковзного вікна (у барах) для порогу
                верхнього терциля (за замовчуванням
                config.settings.MOMENTUM_TERCILE_WINDOW). Фіксований —
                НЕ підбирається під дані.
            tercile_pct: квантиль порогу верхнього терциля, у частках
                (за замовчуванням config.settings.TERCILE_PCT = 0.667).
            name: ім'я стратегії для логування й іменування файлів звіту
                (успадковано від Strategy).

        Викидає:
            ValueError: якщо lookback чи tercile_window <= 0, або
                tercile_pct поза межами (0, 1).
        """
        super().__init__(name=name)

        if lookback <= 0:
            raise ValueError(f"lookback має бути додатним, отримано {lookback!r}.")
        if tercile_window <= 0:
            raise ValueError(f"tercile_window має бути додатним, отримано {tercile_window!r}.")
        if not (0.0 < tercile_pct < 1.0):
            raise ValueError(f"tercile_pct має бути в межах (0, 1), отримано {tercile_pct!r}.")

        self.lookback = lookback
        self.tercile_window = tercile_window
        self.tercile_pct = tercile_pct

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерує сигнали на основі 30-денного моментуму з ковзним терциль-порогом.

        Параметри:
            df: DataFrame з колонкою 'close', відсортований за часом за
                зростанням (найстаріший бар перший).

        Повертає:
            Копію df з доданими колонками:
                momentum_return: log-дохідність close-to-close за lookback барів.
                momentum_tercile_threshold: ковзний поріг верхнього терциля,
                    порахований БЕЗ поточного бару (NaN на перших
                    lookback + tercile_window барах, поки попереднє вікно
                    [t-tercile_window, t-1] не набрало повний розмір).
                momentum_state_active: bool, чи дохідність вище порогу.
                signal: 1 / -1 / 0.

        Викидає:
            KeyError: якщо в df відсутня колонка 'close'.
        """
        if "close" not in df.columns:
            raise KeyError(
                "У df відсутня обов'язкова колонка 'close' для Momentum30Strategy."
            )

        df = df.copy()
        close = df["close"]

        # log-дохідність close-to-close за lookback барів. Значення на барі t
        # використовує лише close[t] і close[t-lookback] — обидва вже відомі
        # на момент закриття бару t.
        momentum_return = np.log(close / close.shift(self.lookback))

        # КРИТИЧНО: поріг для бару t рахується з БАРІВ [t - tercile_window, t - 1],
        # БЕЗ самого t. .shift(1) ЗАСТОСОВАНО ПЕРЕД .rolling(): це зсуває всю
        # серію на один бар назад, тож вікно .rolling(), яке саме по собі
        # дивиться на останні tercile_window точок ЗІ зсунутої серії, що
        # закінчуються на позиції t, фактично складається з ОРИГІНАЛЬНИХ
        # значень momentum_return у позиціях [t - tercile_window, t - 1].
        # Без цього .shift(1) момент t потрапляв би у власне вікно порогу
        # (не lookahead у класичному сенсі — жодних барів ПІСЛЯ t, — але
        # значення брало б участь у визначенні порогу, який його ж
        # класифікує; саме це виправлено тут).
        # min_periods=tercile_window: поріг залишається NaN (-> state=False),
        # поки попереднє вікно не набрало повний розмір — так само, як
        # EMA-200 у ema_cross.py "розігрівається" перші 200 барів.
        momentum_threshold = (
            momentum_return.shift(1)
            .rolling(window=self.tercile_window, min_periods=self.tercile_window)
            .quantile(self.tercile_pct)
        )

        # Порівняння з NaN (розігрівний період) дає False — коректний "немає
        # сигналу" стан без додаткових перевірок.
        state_active = momentum_return > momentum_threshold

        # .shift(1) на bool-серії дає NaN на першому барі (немає попереднього
        # стану) — fill_value=False одразу приводить перший бар до
        # коректного bool без проміжного object-dtype/NaN у & / ~ нижче.
        state_prev = state_active.shift(1, fill_value=False)

        entry = state_active & ~state_prev
        reverse_exit = ~state_active & state_prev

        df["momentum_return"] = momentum_return
        df["momentum_tercile_threshold"] = momentum_threshold
        df["momentum_state_active"] = state_active
        df["signal"] = np.select(
            [entry, reverse_exit],
            [1, -1],
            default=0,
        ).astype(int)

        n_entries = int((df["signal"] == 1).sum())
        n_exits = int((df["signal"] == -1).sum())
        n_active_bars = int(state_active.sum())
        logger.info(
            "%s: згенеровано %d вхідних і %d зворотних сигналів на %d барах "
            "(активний стан на %d барах, lookback=%d, tercile_window=%d, tercile_pct=%.3f).",
            self.name, n_entries, n_exits, len(df), n_active_bars,
            self.lookback, self.tercile_window, self.tercile_pct,
        )

        return df
