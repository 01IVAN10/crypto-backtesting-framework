"""
Модуль завантаження історичних OHLCV-даних з біржі через ccxt.

Відповідальність цього модуля — лише завантаження й збереження сирих даних.
Розрахунок індикаторів та будь-яка подальша обробка живуть в indicators.py.

Ключові моменти реалізації:
- Binance (як і більшість бірж через ccxt) віддає максимум ~1000 свічок за
  один виклик fetch_ohlcv, тому для повної історії потрібна пагінація:
  кожен наступний запит стартує з часу останньої отриманої свічки.
- Для публічних історичних даних (OHLCV) API-ключі Binance не потрібні —
  клієнт створюється без авторизації.
- Мережеві помилки та rate limit біржі обробляються повторними спробами
  з експоненційною затримкою.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

from bot.utils.logger import get_logger

logger = get_logger(__name__)

# Максимальна кількість свічок за один запит fetch_ohlcv (ліміт Binance)
MAX_CANDLES_PER_REQUEST = 1000
# Пауза між послідовними запитами (додатково до вбудованого enableRateLimit ccxt)
REQUEST_PAUSE_SECONDS = 0.2
# Параметри повторних спроб при мережевих помилках / rate limit
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0


def create_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    """
    Створює клієнт біржі ccxt для завантаження публічних даних.

    :param exchange_id: ідентифікатор біржі в ccxt (напр. "binance")
    :return: ініціалізований екземпляр ccxt.Exchange
    :raises AttributeError: якщо exchange_id не підтримується ccxt
    """
    exchange_class = getattr(ccxt, exchange_id)
    exchange: ccxt.Exchange = exchange_class({"enableRateLimit": True})
    logger.info("Створено клієнт біржі: %s", exchange_id)
    return exchange


def _date_to_timestamp_ms(date_str: str) -> int:
    """Перетворює дату 'YYYY-MM-DD' (UTC) на unix-timestamp у мілісекундах."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _fetch_with_retry(
    exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int
) -> list[list[float]]:
    """
    Виконує один виклик fetch_ohlcv з повторними спробами при тимчасових помилках.

    :raises ccxt.ExchangeError: при неповторюваній помилці біржі (напр. невірний символ)
    :raises RuntimeError: якщо всі спроби вичерпано через мережеві помилки/rate limit
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=MAX_CANDLES_PER_REQUEST,
            )
        except ccxt.RateLimitExceeded as exc:
            wait = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Перевищено ліміт запитів (спроба %d/%d): %s. Чекаємо %.1fс",
                attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
        except ccxt.NetworkError as exc:
            wait = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Мережева помилка (спроба %d/%d): %s. Чекаємо %.1fс",
                attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
        except ccxt.ExchangeError as exc:
            # Неповторювана помилка біржі (напр. невірний символ/параметри) — пробросимо одразу
            logger.error("Помилка біржі при запиті %s (%s): %s", symbol, timeframe, exc)
            raise

    raise RuntimeError(
        f"Не вдалося завантажити дані для {symbol} ({timeframe}) після {MAX_RETRIES} спроб"
    )


def _candles_to_dataframe(candles: list[list[float]]) -> pd.DataFrame:
    """Перетворює список сирих свічок ccxt [ts, o, h, l, c, v] на pandas DataFrame."""
    df = pd.DataFrame(
        candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["timestamp", "datetime", "open", "high", "low", "close", "volume"]]


def fetch_ohlcv_full_history(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_date: str,
) -> pd.DataFrame:
    """
    Завантажує повну історію OHLCV-свічок для символу від since_date до поточного моменту.

    Виконує пагінацію запитів: кожен наступний виклик fetch_ohlcv стартує з часу
    останньої отриманої свічки + 1 таймфрейм, доки не буде досягнуто поточний час.

    :param exchange: клієнт біржі ccxt (див. create_exchange)
    :param symbol: торгова пара, напр. "BTC/USDT"
    :param timeframe: таймфрейм свічок, напр. "1h", "4h", "1d"
    :param since_date: дата початку завантаження у форматі "YYYY-MM-DD" (UTC)
    :return: DataFrame з колонками [timestamp, datetime, open, high, low, close, volume],
             відсортований за часом, без дублікатів
    :raises ValueError: якщо symbol відсутній на біржі
    """
    markets = exchange.load_markets()
    if symbol not in markets:
        raise ValueError(f"Символ {symbol} відсутній на біржі {exchange.id}")

    since_ms = _date_to_timestamp_ms(since_date)
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    now_ms = exchange.milliseconds()

    all_candles: list[list[float]] = []
    current_since = since_ms
    is_first_batch = True

    logger.info("Починаємо завантаження %s (%s) з %s", symbol, timeframe, since_date)

    while current_since < now_ms:
        candles = _fetch_with_retry(exchange, symbol, timeframe, current_since)

        if not candles:
            break

        if is_first_batch:
            # Звіряємо запитану дату початку з тим, що фактично віддала біржа:
            # якщо на since_date реальної торгівлі ще не було, Binance поверне
            # найранішу доступну свічку, а не порожній результат.
            actual_first_ts = candles[0][0]
            actual_first_date = datetime.fromtimestamp(actual_first_ts / 1000, tz=timezone.utc).date()
            requested_date = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).date()
            if actual_first_date != requested_date:
                logger.warning(
                    "%s (%s): запитана дата початку %s недоступна на біржі. "
                    "Найраніша фактично доступна дата: %s — саме з неї починається завантаження.",
                    symbol, timeframe, requested_date, actual_first_date,
                )
            else:
                logger.info(
                    "%s (%s): підтверджено — дані доступні саме з запитаної дати %s",
                    symbol, timeframe, requested_date,
                )
            is_first_batch = False

        all_candles.extend(candles)
        last_candle_ts = candles[-1][0]
        next_since = last_candle_ts + timeframe_ms

        # Захист від нескінченного циклу, якщо біржа раптом повертає ту саму свічку знову
        if next_since <= current_since:
            break
        current_since = next_since

        logger.info(
            "  отримано %d свічок (всього: %d), поточна дата: %s",
            len(candles),
            len(all_candles),
            datetime.fromtimestamp(last_candle_ts / 1000, tz=timezone.utc).date(),
        )

        time.sleep(REQUEST_PAUSE_SECONDS)

    df = _candles_to_dataframe(all_candles)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    logger.info("Завершено: %s (%s) — %d рядків", symbol, timeframe, len(df))
    return df


def save_raw_data(df: pd.DataFrame, symbol: str, timeframe: str, raw_data_dir: Path) -> Path:
    """
    Зберігає DataFrame у CSV в raw_data_dir.

    Ім'я файлу: {SYMBOL}_{timeframe}.csv, де "/" в символі замінено на "_"
    (напр. "BTC/USDT" + "1h" -> "BTC_USDT_1h.csv").

    :return: шлях до збереженого файлу
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
    filepath = raw_data_dir / filename
    df.to_csv(filepath, index=False)
    logger.info("Збережено: %s (%d рядків)", filepath, len(df))
    return filepath


def download_symbol_data(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_date: str,
    raw_data_dir: Path,
) -> Path:
    """
    Завантажує та зберігає історичні дані для одного символу й таймфрейму.

    Обгортка над fetch_ohlcv_full_history + save_raw_data для зручного виклику
    зі скрипта scripts/download_data.py.

    :return: шлях до збереженого CSV-файлу
    """
    df = fetch_ohlcv_full_history(exchange, symbol, timeframe, since_date)
    return save_raw_data(df, symbol, timeframe, raw_data_dir)