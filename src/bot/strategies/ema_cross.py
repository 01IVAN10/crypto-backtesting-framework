"""
src/bot/strategies/ema_cross.py

Baseline-стратегія EMA(20/50) crossover з трендовим фільтром EMA(200) —
перший тестовий стенд для перевірки коректності двигуна бектесту
(НЕ фінальна стратегія, див. docs/trading_decisions.md, розділ "Baseline").
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bot.strategies.base import Strategy
from config import settings

logger = logging.getLogger(__name__)


class EMACrossStrategy(Strategy):
    """
    EMA(short/medium) crossover з трендовим фільтром EMA(long).

    Правила сигналів:
        1 (вхід у лонг): EMA(short) перетинає EMA(medium) знизу вгору
            І close на цьому ж барі вище EMA(long).
        -1 (зворотний сигнал / вихід): EMA(short) перетинає EMA(medium)
            згори вниз. Трендовий фільтр EMA(long) на вихід НЕ впливає —
            вихід за зворотним сигналом має спрацьовувати незалежно від
            того, де зараз ціна відносно EMA(long).
        0: решта барів.

    Тільки лонг, шортів немає (config.settings.DIRECTION == "long_only").

    Очікує, що df вже містить готові колонки індикаторів, пораховані в
    data_pipeline/indicators.py (тут вони НЕ перераховуються):
        ema_{ema_short_period}, ema_{ema_medium_period}, ema_{ema_long_period}
    За замовчуванням (config.settings) це ema_20, ema_50, ema_200.

    Перетин визначається порівнянням поточного бару з попереднім
    (.shift(1)) — сигнал формується на закритті бару, жодні дані
    наступних барів не використовуються (без lookahead bias).
    """

    def __init__(
        self,
        ema_short_period: int = settings.EMA_SHORT_PERIOD,
        ema_medium_period: int = settings.EMA_MEDIUM_PERIOD,
        ema_long_period: int = settings.EMA_LONG_PERIOD,
        name: str = "ema_cross",
    ) -> None:
        """
        Ініціалізує стратегію.

        Параметри:
            ema_short_period: період короткої EMA
                (за замовчуванням config.settings.EMA_SHORT_PERIOD).
            ema_medium_period: період середньої EMA
                (за замовчуванням config.settings.EMA_MEDIUM_PERIOD).
            ema_long_period: період довгої EMA — трендовий фільтр
                (за замовчуванням config.settings.EMA_LONG_PERIOD).
            name: ім'я стратегії для логування (успадковано від Strategy).

        Викидає:
            ValueError: якщо періоди не утворюють коректний порядок
                short < medium < long, або якщо будь-який період <= 0.
        """
        super().__init__(name=name)

        if ema_short_period <= 0 or ema_medium_period <= 0 or ema_long_period <= 0:
            raise ValueError(
                f"Періоди EMA мають бути додатними: "
                f"short={ema_short_period}, medium={ema_medium_period}, long={ema_long_period}."
            )
        if not (ema_short_period < ema_medium_period < ema_long_period):
            raise ValueError(
                f"Періоди EMA мають бути впорядковані short < medium < long, "
                f"отримано short={ema_short_period}, medium={ema_medium_period}, long={ema_long_period}."
            )

        self.ema_short_period = ema_short_period
        self.ema_medium_period = ema_medium_period
        self.ema_long_period = ema_long_period

        self._col_short = f"ema_{ema_short_period}"
        self._col_medium = f"ema_{ema_medium_period}"
        self._col_long = f"ema_{ema_long_period}"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерує сигнали на основі готових колонок EMA.

        Параметри:
            df: DataFrame з колонкою close і колонками ema_{short}/ema_{medium}/ema_{long},
                відсортований за часом за зростанням (найстаріший бар перший).

        Повертає:
            Копію df з доданою колонкою 'signal' (int: 1 / -1 / 0).

        Викидає:
            KeyError: якщо в df відсутня хоч одна з обов'язкових колонок
                (close, ema_{short}, ema_{medium}, ema_{long}).
        """
        required_columns = ["close", self._col_short, self._col_medium, self._col_long]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise KeyError(
                f"У df відсутні обов'язкові колонки для EMACrossStrategy: {missing}. "
                f"Переконайся, що data_pipeline/indicators.py порахував EMA "
                f"для періодів {self.ema_short_period}/{self.ema_medium_period}/{self.ema_long_period} "
                f"перед викликом generate_signals()."
            )

        df = df.copy()

        ema_short = df[self._col_short]
        ema_medium = df[self._col_medium]
        close = df["close"]
        ema_long = df[self._col_long]

        # Порівняння виключно з попереднім баром (.shift(1)) — рішення
        # приймається на закритті поточного бару, дані майбутніх барів
        # у розрахунок не потрапляють.
        ema_short_prev = ema_short.shift(1)
        ema_medium_prev = ema_medium.shift(1)

        cross_up = (ema_short_prev <= ema_medium_prev) & (ema_short > ema_medium)
        cross_down = (ema_short_prev >= ema_medium_prev) & (ema_short < ema_medium)

        trend_filter = close > ema_long

        long_entry = cross_up & trend_filter
        reverse_exit = cross_down

        # NaN у "розігрівних" барах (перші ~ema_long_period рядків, поки
        # ковзні середні ще не набрали вікно) дають False у порівняннях,
        # тож на цих барах сигнал коректно лишається 0.
        df["signal"] = np.select(
            [long_entry, reverse_exit],
            [1, -1],
            default=0,
        ).astype(int)

        n_entries = int((df["signal"] == 1).sum())
        n_exits = int((df["signal"] == -1).sum())
        logger.info(
            "%s: згенеровано %d вхідних і %d зворотних сигналів на %d барах.",
            self.name, n_entries, n_exits, len(df),
        )

        return df