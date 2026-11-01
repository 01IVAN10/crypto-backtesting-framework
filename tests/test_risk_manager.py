"""
tests/test_risk_manager.py

Тести для src/bot/risk/risk_manager.py.

Ключові сценарії:
- вузький ATR-стоп (тихий період) коректно тригерить СТЕЛЮ 50% балансу,
  а не якийсь прихований floor, і stop_distance/stop_loss_price лишаються
  фіксованими незалежно від обрізання розміру;
- малий розрахований нотіонал коректно призводить до skip (без округлення
  вгору до мінімуму біржі);
- відмова при вже відкритій позиції;
- валідація некоректних вхідних даних.

Тейк-профіту в risk_manager немає (прибрано разом з ATR_TP_MULT —
верхню межу угоди тепер формує трейлінг-стоп у backtesting/engine.py).
"""

import pytest

from bot.risk.risk_manager import (
    PositionSizingError,
    calculate_position_size,
    extract_min_notional_from_market,
)
from config import settings


def test_narrow_atr_stop_triggers_ceiling_not_floor() -> None:
    """
    Тихий період -> малий ATR -> вузький стоп -> базовий розмір
    (balance × risk_pct / stop_distance) виходить величезним у нотіоналі.

    Очікуємо: спрацьовує СТЕЛЯ max_position_pct балансу (єдиний запобіжник
    розміру), а не неявний floor на ширині стопу. stop_distance/stop_loss_price
    залишаються фіксованими незалежно від обрізання розміру.
    """
    balance = 150.0
    entry_price = 30_000.0
    atr = 1.0  # дуже вузький ATR -> stop_distance = ATR_STOP_MULT * atr = 2 USDT

    result = calculate_position_size(
        balance=balance,
        entry_price=entry_price,
        atr=atr,
        has_open_position=False,
        min_notional=5.0,
    )

    # Переконуємось, що тест дійсно перевіряє стелю, а не мінімальний нотіонал:
    # базовий (необрізаний) нотіонал має бути набагато більшим за стелю.
    raw_size = (balance * settings.RISK_PCT) / (settings.ATR_STOP_MULT * atr)
    raw_notional = raw_size * entry_price
    max_notional = balance * settings.MAX_POSITION_PCT
    assert raw_notional > max_notional

    assert result.approved is True
    assert result.notional == pytest.approx(max_notional)
    assert result.position_size == pytest.approx(max_notional / entry_price)

    # stop_distance і ціна стопу НЕ залежать від обрізання розміру стелею.
    expected_stop_distance = settings.ATR_STOP_MULT * atr
    assert result.stop_distance == pytest.approx(expected_stop_distance)
    assert result.stop_loss_price == pytest.approx(entry_price - expected_stop_distance)


def test_below_min_notional_is_skipped_not_rounded_up() -> None:
    """
    Широкий ATR-стоп + малий баланс -> розрахований нотіонал менший за
    мінімум біржі. Угода має бути пропущена (approved=False), розмір
    НЕ повинен бути примусово округлений до мінімуму.
    """
    balance = 50.0
    entry_price = 10_000.0
    atr = 1_500.0  # широкий ATR -> малий розмір позиції в BTC
    min_notional = 5.0

    # Переконуємось, що тест дійсно перевіряє min_notional, а не стелю:
    # базовий нотіонал має бути одночасно меншим за стелю і за min_notional.
    raw_size = (balance * settings.RISK_PCT) / (settings.ATR_STOP_MULT * atr)
    raw_notional = raw_size * entry_price
    assert raw_notional < balance * settings.MAX_POSITION_PCT
    assert raw_notional < min_notional

    result = calculate_position_size(
        balance=balance,
        entry_price=entry_price,
        atr=atr,
        has_open_position=False,
        min_notional=min_notional,
    )

    assert result.approved is False
    assert result.reason == "below_min_notional"
    assert result.position_size is None
    assert result.notional is None
    # Стоп-лос все одно розрахований — для логування причини відмови.
    assert result.stop_loss_price == pytest.approx(entry_price - settings.ATR_STOP_MULT * atr)


def test_open_position_is_rejected_explicitly() -> None:
    """Максимум одна відкрита позиція одночасно — функція має явно відмовити."""
    result = calculate_position_size(
        balance=150.0,
        entry_price=30_000.0,
        atr=500.0,
        has_open_position=True,
        min_notional=5.0,
    )

    assert result.approved is False
    assert result.reason == "already_open_position"
    assert result.position_size is None
    assert result.notional is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"balance": 0.0},
        {"balance": -10.0},
        {"entry_price": 0.0},
        {"entry_price": -100.0},
        {"atr": 0.0},
        {"atr": -1.0},
        {"min_notional": -1.0},
    ],
)
def test_invalid_inputs_raise_error(overrides: dict) -> None:
    """Некоректні вхідні дані (<= 0 там, де це недопустимо) викликають PositionSizingError."""
    base_kwargs = {
        "balance": 150.0,
        "entry_price": 30_000.0,
        "atr": 500.0,
        "has_open_position": False,
        "min_notional": 5.0,
    }
    base_kwargs.update(overrides)

    with pytest.raises(PositionSizingError):
        calculate_position_size(**base_kwargs)


def test_extract_min_notional_from_market_uses_fallback_when_missing() -> None:
    """Якщо ccxt-market не містить ліміту limits.cost.min — використовується fallback."""
    market_without_limit = {"limits": {"cost": {"min": None}}}
    assert extract_min_notional_from_market(market_without_limit, fallback=7.5) == 7.5


def test_extract_min_notional_from_market_reads_ccxt_limit() -> None:
    """Якщо ccxt повернув ліміт — використовується саме він, а не fallback."""
    market = {"limits": {"cost": {"min": 10.0}}}
    assert extract_min_notional_from_market(market, fallback=5.0) == 10.0