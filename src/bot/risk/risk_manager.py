"""
src/bot/risk/risk_manager.py

Розрахунок розміру позиції та початкового стоп-лосу для спотової довгої
(long-only) торгівлі. Тейк-профіту тут НЕМАЄ — верхню межу угоди формує
трейлінг-стоп у backtesting/engine.py (перепроектування виходів за
результатами MFE-заміру, docs/trading_decisions.md), а не фіксований
рівень ATR. Це вихідна точка перед двигуном бектесту: engine викликає
calculate_position_size() на кожному сигналі входу і діє за полем approved
у результаті.

Навмисне архітектурне рішення: модуль НЕ звертається до біржі напряму
(жодних викликів ccxt всередині). Мінімальний нотіонал (min_notional)
передається як звичайний аргумент-число, а не тягнеться тут із
exchange.load_markets(). Це робить функцію чистою (без мережі, без
побічних ефектів) і легко тестованою — виклики exchange.load_markets()
лишаються на шарі вище (execution/backtesting), який передає готове
число сюди. Для парсингу відповіді ccxt є окрема допоміжна функція
extract_min_notional_from_market().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


class PositionSizingError(Exception):
    """Некоректні вхідні дані для розрахунку розміру позиції."""


@dataclass(frozen=True)
class PositionSizingResult:
    """
    Результат розрахунку position sizing.

    Атрибути:
        approved: чи угода схвалена до відкриття.
        reason: причина відмови, якщо approved=False ("already_open_position"
            або "below_min_notional"); інакше None.
        position_size: розмір позиції в базовому активі (напр. BTC).
            None, якщо угода не схвалена.
        notional: вартість позиції в котирувальній валюті (USDT).
            None, якщо угода не схвалена.
        stop_loss_price: ціна ПОЧАТКОВОГО стоп-лосу. Розраховується завжди,
            незалежно від approved — потрібна для логування/діагностики
            відмов. Верхньої межі (тейк-профіту) тут немає — її формує
            трейлінг-стоп у backtesting/engine.py, не risk_manager.
        stop_distance: відстань від входу до стопу в ціновому вираженні
            (atr_stop_mult × ATR). Не залежить від обрізання розміру стелею.
        risk_amount: сума ризику в котирувальній валюті (balance × risk_pct).
    """

    approved: bool
    reason: str | None
    position_size: float | None
    notional: float | None
    stop_loss_price: float
    stop_distance: float
    risk_amount: float


def calculate_position_size(
    balance: float,
    entry_price: float,
    atr: float,
    has_open_position: bool,
    min_notional: float,
    risk_pct: float = settings.RISK_PCT,
    max_position_pct: float = settings.MAX_POSITION_PCT,
    atr_stop_mult: float = settings.ATR_STOP_MULT,
) -> PositionSizingResult:
    """
    Розраховує розмір позиції та початковий стоп-лос для входу в лонг.

    Тейк-профіту тут немає: верхню межу угоди формує трейлінг-стоп у
    backtesting/engine.py (перепроектування виходів за MFE-заміром), а не
    фіксований рівень ATR тут.

    Порядок кроків фіксований і навмисно НЕ змінюється залежно від
    результату попереднього кроку (без прихованих floor-перевизначень):

    1. Стоп-лос рахується ЗАВЖДИ як entry_price − atr_stop_mult × ATR.
    2. Якщо вже є відкрита позиція — угода відмовляється явно
       (reason="already_open_position"), максимум одна позиція одночасно.
    3. Базовий розмір позиції = (balance × risk_pct) / stop_distance.
    4. Якщо нотіонал базового розміру перевищує стелю
       (balance × max_position_pct) — розмір ОБРІЗАЄТЬСЯ до стелі.
       Це єдиний запобіжник розміру: stop_distance і ціна стопу
       не чіпаються.
    5. Якщо підсумковий нотіонал менший за мінімальний нотіонал біржі —
       угода пропускається (reason="below_min_notional"). Розмір НЕ
       округляється вгору до мінімуму.

    Параметри:
        balance: поточний баланс рахунку в котирувальній валюті (USDT).
        entry_price: ціна входу в позицію.
        atr: значення ATR(14) на момент входу, в тих самих одиницях, що й ціна.
        has_open_position: чи вже є відкрита позиція (ліміт — одна одночасно).
        min_notional: мінімальний дозволений розмір ордера в котирувальній
            валюті. Бери з exchange.load_markets()[symbol]['limits']['cost']['min']
            (див. extract_min_notional_from_market) або з
            config.settings.DEFAULT_MIN_NOTIONAL як fallback у тестах.
        risk_pct: частка балансу, якою ризикуємо за угоду
            (за замовчуванням — config.settings.RISK_PCT).
        max_position_pct: стеля розміру позиції як частка балансу
            (за замовчуванням — config.settings.MAX_POSITION_PCT).
        atr_stop_mult: множник ATR для стоп-лосу
            (за замовчуванням — config.settings.ATR_STOP_MULT).

    Повертає:
        PositionSizingResult з рішенням approved/reason і розрахованими
        величинами (стоп-лос рахується завжди, навіть при відмові — для логів).

    Викидає:
        PositionSizingError: якщо вхідні дані некоректні (balance,
            entry_price або atr <= 0; min_notional < 0; risk_pct чи
            max_position_pct поза межами (0, 1]; atr_stop_mult <= 0).
    """
    _validate_inputs(
        balance=balance,
        entry_price=entry_price,
        atr=atr,
        min_notional=min_notional,
        risk_pct=risk_pct,
        max_position_pct=max_position_pct,
        atr_stop_mult=atr_stop_mult,
    )

    # Крок 1: стоп-лос рахується завжди, до будь-яких перевірок і обрізань.
    stop_distance = atr_stop_mult * atr
    stop_loss_price = entry_price - stop_distance
    risk_amount = balance * risk_pct

    # Крок 2: максимум одна відкрита позиція.
    if has_open_position:
        logger.info("Відмова у відкритті позиції: вже є відкрита позиція (ліміт — 1).")
        return PositionSizingResult(
            approved=False,
            reason="already_open_position",
            position_size=None,
            notional=None,
            stop_loss_price=stop_loss_price,
            stop_distance=stop_distance,
            risk_amount=risk_amount,
        )

    # Крок 3: базовий розмір позиції з ризику на угоду.
    raw_size = risk_amount / stop_distance
    raw_notional = raw_size * entry_price

    # Крок 4: єдиний запобіжник розміру — стеля % балансу.
    max_notional = balance * max_position_pct
    if raw_notional > max_notional:
        notional = max_notional
        position_size = max_notional / entry_price
        logger.info(
            "Розмір позиції обрізано стелею %.1f%% балансу: %.8f -> %.8f "
            "(нотіонал %.2f -> %.2f). Стоп-лос не змінився.",
            max_position_pct * 100, raw_size, position_size, raw_notional, notional,
        )
    else:
        notional = raw_notional
        position_size = raw_size

    # Крок 5: мінімальний нотіонал біржі — без округлення вгору.
    if notional < min_notional:
        logger.info(
            "Угода пропущена (skip): нотіонал %.2f менший за мінімальний нотіонал біржі %.2f.",
            notional, min_notional,
        )
        return PositionSizingResult(
            approved=False,
            reason="below_min_notional",
            position_size=None,
            notional=None,
            stop_loss_price=stop_loss_price,
            stop_distance=stop_distance,
            risk_amount=risk_amount,
        )

    logger.info(
        "Позиція схвалена: розмір %.8f (нотіонал %.2f USDT), SL=%.2f.",
        position_size, notional, stop_loss_price,
    )
    return PositionSizingResult(
        approved=True,
        reason=None,
        position_size=position_size,
        notional=notional,
        stop_loss_price=stop_loss_price,
        stop_distance=stop_distance,
        risk_amount=risk_amount,
    )


def extract_min_notional_from_market(
    market: dict,
    fallback: float = settings.DEFAULT_MIN_NOTIONAL,
) -> float:
    """
    Дістає мінімальний нотіонал ордера зі структури ринку ccxt
    (результат exchange.load_markets()[symbol]).

    Сама функція не робить мережевих викликів — очікує вже завантажений
    словник market від виклику, зробленого шаром execution/backtesting.

    Параметри:
        market: словник ринку ccxt, зокрема market['limits']['cost']['min'].
        fallback: значення, яке повертається, якщо біржа не надала ліміт
            (за замовчуванням — config.settings.DEFAULT_MIN_NOTIONAL).

    Повертає:
        Мінімальний нотіонал у котирувальній валюті.

    Викидає:
        PositionSizingError: якщо market має несподівану структуру
            (наприклад, market — не словник).
    """
    try:
        limit = market.get("limits", {}).get("cost", {}).get("min")
    except AttributeError as exc:
        raise PositionSizingError(f"Некоректна структура market від ccxt: {exc}") from exc

    if limit is None:
        logger.warning(
            "Біржа не повернула мінімальний нотіонал (limits.cost.min) для ринку, "
            "використовую fallback=%.2f.",
            fallback,
        )
        return fallback

    return float(limit)


def _validate_inputs(
    balance: float,
    entry_price: float,
    atr: float,
    min_notional: float,
    risk_pct: float,
    max_position_pct: float,
    atr_stop_mult: float,
) -> None:
    """Перевіряє коректність вхідних параметрів перед розрахунком розміру позиції."""
    if balance <= 0:
        raise PositionSizingError(f"balance має бути додатним, отримано {balance!r}.")
    if entry_price <= 0:
        raise PositionSizingError(f"entry_price має бути додатним, отримано {entry_price!r}.")
    if atr <= 0:
        raise PositionSizingError(f"atr має бути додатним, отримано {atr!r}.")
    if min_notional < 0:
        raise PositionSizingError(f"min_notional не може бути від'ємним, отримано {min_notional!r}.")
    if not (0 < risk_pct <= 1):
        raise PositionSizingError(f"risk_pct має бути в межах (0, 1], отримано {risk_pct!r}.")
    if not (0 < max_position_pct <= 1):
        raise PositionSizingError(
            f"max_position_pct має бути в межах (0, 1], отримано {max_position_pct!r}."
        )
    if atr_stop_mult <= 0:
        raise PositionSizingError(f"atr_stop_mult має бути додатним, отримано {atr_stop_mult!r}.")