"""
Централізовані налаштування проекту: тікери, таймфрейми, шляхи до даних/логів.

Правило проекту: параметри, що часто змінюються, живуть тут, а не хардкодяться
в скриптах чи модулях src/bot/.
"""

from __future__ import annotations

from pathlib import Path

# --- Кореневий каталог проекту (crypto_bot/) ---
# settings.py лежить у config/, тому корінь — на один рівень вище
ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Шляхи до даних ---
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- Каталог логів ---
LOGS_DIR = ROOT_DIR / "logs"

# --- Каталог результатів бектесту (журнали угод, звіти) ---
RESULTS_DIR = ROOT_DIR / "results"

# --- Біржа (ccxt exchange id) ---
EXCHANGE_ID = "binance"

# --- Тікери для завантаження та обробки ---
# Формат ccxt: "БАЗА/КОТИРУВАННЯ", напр. "BTC/USDT"
# ETH/USDT додається пізніше — як другий набір даних для перевірки робастності
# (див. docs/trading_decisions.md), поки лише BTC.
SYMBOLS: list[str] = [
    "BTC/USDT",
]

# --- Таймфрейми свічок ---
# 4h обрано свідомо: на 1h кількість угод у 3-4 рази більша, комісії (0.2% за круг)
# з'їдають edge; на 1d замало угод для статистично значущої вибірки.
# 1d додано для статистичної потужності: на 1d за 2021-2023 лише ~1095 барів,
# після врахування перекриття вікон n_eff падає до кількох десятків — замало.
TIMEFRAMES: list[str] = [
    "4h",
    "1d",
]

# --- Дата початку завантаження історії (формат YYYY-MM-DD, UTC) ---
# 2017-08-17 — дата старту торгів BTC/USDT на Binance. Якщо біржа фактично
# віддає дані з пізнішої дати, downloader.py логує найранішу отриману дату —
# звіряй з цим коментарем, якщо помітиш розбіжність.
# УВАГА: 2017-2020 — структурно інший ринок (менша ліквідність, менше
# інституційних грошей), ніж 2021+. Дані беремо повністю, але це треба
# враховувати при інтерпретації статистик з цього періоду.
DOWNLOAD_SINCE_DATE = "2017-08-17"

# --- Межі розбиття processed-даних на exploration/holdout (splitter.py) ---
# Exploration: усе до і включно EXPLORATION_END_DATE — тут відбувається EDA
# та підбір логіки стратегії.
# Holdout: усе від HOLDOUT_START_DATE і далі — НЕ використовується під час
# дослідження, лише для фінальної одноразової перевірки (запобігання overfitting).
EXPLORATION_END_DATE = "2023-12-31"
HOLDOUT_START_DATE = "2024-01-01"

# --- Ризик-менеджмент (position sizing, src/bot/risk/risk_manager.py) ---
# Джерело правди — docs/trading_decisions.md, розділ "Ризик-менеджмент".
RISK_PCT: float = 0.02        # частка балансу, якою ризикуємо за одну угоду
ATR_STOP_MULT: float = 2.0    # початковий стоп-лос = entry_price − ATR_STOP_MULT × ATR(14)-на-вході
MAX_POSITION_PCT: float = 0.5  # стеля розміру позиції як частка балансу — єдиний запобіжник розміру
MAX_OPEN_POSITIONS: int = 1    # максимум одночасно відкритих позицій
EMA_SHORT_PERIOD: int = 20
EMA_MEDIUM_PERIOD: int = 50
EMA_LONG_PERIOD: int = 200

# Мінімальний нотіонал ордера (USDT), використовується ЛИШЕ як fallback,
# коли exchange.load_markets() не повернув ліміт limits.cost.min для пари.
DEFAULT_MIN_NOTIONAL: float = 5.0

# --- Бектест: витрати та капітал (src/bot/backtesting/engine.py) ---
# Джерело правди — docs/trading_decisions.md, розділ "Бектест".
INITIAL_BALANCE: float = 150.0     # стартовий баланс бектесту, USDT — реальний баланс проекту
COMMISSION_PCT: float = 0.001      # комісія біржі 0.1% НА КОЖНУ СТОРОНУ (taker, без знижки BNB)
BASE_SLIPPAGE_PCT: float = 0.0005  # базовий рівень slippage (0.05%) при "нормальній" волатильності
SLIPPAGE_CAP_MULT: float = 3.0     # кеп множника slippage: ефективний slippage <= BASE_SLIPPAGE_PCT × 3

# Трейлінг-стоп БЕЗ порогу активації (перепроектування виходів за MFE-заміром,
# заміна фіксованого take-profit): на кожному барі
# stop = max(поточний stop, running_max_high_since_entry − TRAIL_ATR_MULT × max(ATR_поточного_бару, entry_atr)).
# Трейл сам перебирає керування, щойно кандидат перевищує initial_stop —
# без окремого прапорця активації. ATR тут — ПОТОЧНОГО бару, з нижньою межею
# entry_atr (дистанція не звужується нижче ATR-на-вході навіть при падінні
# поточної волатильності — інакше стоп підстрибує вгору на звичайному
# відкоті). Рухається
# лише вгору (docs/trading_decisions.md, "Ризик-менеджмент").
TRAIL_ATR_MULT: float = 6.0

# --- Крос-секційний набір монет (крок 7.9.4) ---
# Використовується ТІЛЬКИ для крос-секційного заміру (не для 4h-пайплайна
# single-asset EMA-crossover стратегії з BTC/USDT вище).
#
# УВАГА — SURVIVORSHIP BIAS: набір складено з монет, що торгуються СЬОГОДНІ.
# Монети, що зникли або втратили лістинг на Binance за 2018-2023 (напр. делістинг
# після падіння ліквідності чи регуляторних причин), у ньому відсутні. Це
# систематично ЗАВИЩУЄ результат крос-секційної стратегії — переможці, що вижили,
# перепредставлені, невдахи, що зникли, не потрапляють у вибірку. Прибрати це
# повністю неможливо без історичних списків лістингу Binance (яких немає у
# публічному API), тому просто тримати в голові при інтерпретації результатів
# кроку 7.9.4 і будь-якого бектесту на цьому наборі.
#
# EOS/USDT НЕ видаляти, навіть якщо монета фактично припинила торгуватись
# (на практиці її серія обривається в 2025 — див. STALENESS-перевірку в
# verify_data.py). Монета, що зникла з ринку, має брати участь у ранжуванні
# до дати обриву й потім зникнути з юніверсу — саме так виглядало б реальне
# крос-секційне ранжування наживо. Викидання "мертвих" монет із набору і є
# створенням survivorship bias, а не усуненням його.
#
# Розширено з 10 до 22 монет (додано лістинги 2017-2019: NEO, IOTA, ETC,
# DASH, ZEC, QTUM, VET, ONT, BAT, ZRX, WAVES, ATOM) — 10 монет давали терциль
# лише з ~3 активів, замало для крос-секційного ранжування (шум від 1-2 монет).
# Якщо якась із пар відсутня на Binance або має суттєво коротшу історію —
# download_data.py логує це й пропускає пару, БЕЗ підміни на іншу монету
# (підбір складу під доступність — це теж форма підгонки).
CROSS_SECTIONAL_SYMBOLS: list[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "LTC/USDT",
    "LINK/USDT",
    "XLM/USDT",
    "TRX/USDT",
    "EOS/USDT",
    "NEO/USDT",
    "IOTA/USDT",
    "ETC/USDT",
    "DASH/USDT",
    "ZEC/USDT",
    "QTUM/USDT",
    "VET/USDT",
    "ONT/USDT",
    "BAT/USDT",
    "ZRX/USDT",
    "WAVES/USDT",
    "ATOM/USDT",
]

# Таймфрейм для крос-секційного заміру — лише денний (не 4h, щоб не роздувати
# кількість запитів до біржі й обсяг даних для 22 монет за всю історію).
CROSS_SECTIONAL_TIMEFRAME: str = "1d"


def _build_pipeline_targets() -> list[tuple[str, str]]:
    """
    Об'єднує базовий набір SYMBOLS x TIMEFRAMES (single-asset стратегія) з
    CROSS_SECTIONAL_SYMBOLS на CROSS_SECTIONAL_TIMEFRAME (крос-секційний замір)
    в один список пар (symbol, timeframe) без дублікатів — саме його
    використовують download_data.py / add_indicators.py / split_data.py /
    verify_data.py замість роздільних вкладених циклів.
    """
    pairs = {(symbol, timeframe) for symbol in SYMBOLS for timeframe in TIMEFRAMES}
    pairs |= {(symbol, CROSS_SECTIONAL_TIMEFRAME) for symbol in CROSS_SECTIONAL_SYMBOLS}
    return sorted(pairs)


# Повний список (symbol, timeframe), який фактично обробляють скрипти пайплайна.
PIPELINE_TARGETS: list[tuple[str, str]] = _build_pipeline_targets()

# --- Стратегії: Momentum-30 (src/bot/strategies/momentum.py) ---
# Перша НЕ-тестова стратегія (пройшла вимір придатності, крок 7.9.3: 4
# значущі зсуви з 18). Ознака: log-дохідність close-to-close за
# MOMENTUM_LOOKBACK попередніх барів. Стан активний, коли ця дохідність
# у верхньому терцилі розподілу за MOMENTUM_TERCILE_WINDOW попередніх барів
# (ковзне вікно, ЛИШЕ минулі дані — поріг по всій вибірці був би lookahead).
MOMENTUM_LOOKBACK: int = 30
MOMENTUM_TERCILE_WINDOW: int = 252  # ~рік торгових днів
TERCILE_PCT: float = 0.667