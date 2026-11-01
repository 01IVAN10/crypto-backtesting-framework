# Crypto Strategy Backtesting Framework

[![tests](https://github.com/01IVAN10/crypto-backtesting-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/01IVAN10/crypto-backtesting-framework/actions/workflows/tests.yml)

A research framework for testing trading hypotheses on Binance spot data honestly. It is built to answer "does this idea survive realistic execution?", not "how do I make the curve look good?".

> **Status:** research and backtesting only. There is no live order execution: `src/bot/execution/` is intentionally empty, and no strategy has passed the pre-registered gate criteria yet.

## What it does

- **Data pipeline.** Downloads OHLCV candles from Binance via `ccxt` (BTC on 4h and 1d, plus a 22-coin daily cross-sectional universe from 2017). Adds indicators (EMA, ATR) and runs data-integrity checks: gaps in the time grid, NaNs after warm-up, stale or delisted series, history start dates.
- **Exploration / holdout split.** Data up to 2023-12-31 is used for research. Data from 2024 onward is a holdout, reserved for a single final check. The backtest scripts refuse to run on holdout files.
- **Event-driven backtest engine.** Long-only spot simulation:
  - signals are formed on the **close** of bar N and executed at the **open** of bar N+1, for both entries and signal-based exits;
  - commission is charged on both sides (0.1% taker);
  - slippage is volatility-dependent and capped;
  - the initial stop is ATR-based, and the trailing stop only ever moves up (its distance never narrows below the ATR at entry);
  - trades below the exchange's minimum notional are skipped and counted, never rounded up.
- **Risk management.** Fixed-fractional position sizing (2% risk per trade) with a position-size cap. This is a pure function with no network calls, so it is fully unit-testable.
- **Diagnostics per trade.** MFE/MAE in ATR units, whether the trailing stop took over, whether the position-size cap was hit, actual realised risk, and holding time.

## Methodology: rules set before looking at results

Every strategy is judged against gate criteria fixed **before** the run:

| Criterion | Threshold |
|---|---|
| Beats buy-and-hold BTC over the same period | required |
| Number of trades | ≥ 30 |
| Max drawdown | < 25% |
| Profit factor | > 1.3 |

If a strategy fails, its parameters are **not** tuned to make it pass. The next step is a new hypothesis, tested separately. Sharpe ratio is reported for reference only and is not a gate, because crypto returns have heavy tails (excess kurtosis ≈ 8).

Every decision, measurement and dead end is logged in [`docs/trading_decisions.md`](docs/trading_decisions.md) (in Ukrainian), including the ones that turned out wrong. Two examples:
- **Stopping rule declared in advance.** After 0 of 18 entry candidates showed a significant MFE−MAE shift on 4h data, the rule was: one more test on daily data, and if that also fails, stop searching for edge in price-derived indicators. No "let's try 8h, then 12h".
- **A subtle lookahead bug found and fixed.** The momentum threshold's rolling window included the current bar, so a value took part in setting its own threshold. It was fixed with `shift(1)` before `rolling()` and verified with a truncation test (0 mismatches across 1,385 bars). The result moved only slightly (PF 4.96 → 4.44), which shows the finding did not depend on the bug.

## Results (exploration set only)

| Strategy | Data | Return | Max DD | Profit factor | Trades | Buy & hold return / DD | Gate |
|---|---|---|---|---|---|---|---|
| EMA cross (engine test bench) | BTC 4h, 2021–2023 | −0.6% | 16.7% | 0.98 | 35 | +46% / 77% | ❌ |
| Momentum-30 | BTC 1d, 2017–2023 | +144% | 12.3% | 4.44 | 58 | +478% / 77% ¹ | ❌ (fails only "beats B&H") |

Momentum-30 enters when the 30-day log return rises into the top tercile of its own trailing one-year distribution, with the threshold computed strictly from **past** bars. It is in the market ~32% of the time. It has a far better risk profile than holding BTC (Calmar 1.22 vs 0.52), but it does not beat buy-and-hold on return, so by the rules it does not pass. Robustness caveats are stated openly: the top 3 trades make up 66% of net P&L, so the result effectively rests on about 10 events. Full reports and trade logs are in [`results/`](results/).

¹ Buy-and-hold measured from the strategy's first possible entry (2018-07-18), since the 282-bar warm-up cannot trade. From the very start of the data (2017-08-17) it is +892% / 83% DD, which is the figure in the report file.

The EMA-cross run predates the extension of the exploration history back to 2017, so re-running `scripts/backtest.py` today uses 2017–2023 data and gives different numbers. The Momentum-30 report reproduces exactly (identical trade log).

The holdout set has not been touched.

## Known limitations

- **Survivorship bias** in the cross-sectional universe: the coins were picked from what trades today. Coins delisted in 2018–2023 are missing, which inflates cross-sectional results. Dead coins that are still in the data (e.g. EOS) are deliberately kept rather than removed.
- 2017–2020 is a structurally different market (lower liquidity, fewer institutions), so statistics from that period are interpreted with care.
- A single asset and a single strategy family, with small samples (tens of trades).
- No live execution, no order-book modelling, and no funding or borrow costs (long-only spot).

## Tests

```bash
pip install -e ".[dev]"
pytest -q        # 46 tests
```

The tests target the places where backtests usually lie:
- entry and signal-based exit happen at the **next** bar's open, not on the signal bar;
- the momentum threshold ignores future data, and truncating the dataframe leaves past signals identical;
- the trailing stop never moves down, takes over without an activation threshold, and does not narrow below the entry ATR;
- commission is charged separately on entry and exit;
- trades below the minimum notional are skipped, not rounded up;
- strategies never mutate their input dataframe.

## Project structure

```
config/settings.py              # all parameters: symbols, dates, risk, costs, strategy params
src/bot/data_pipeline/          # downloader, indicators, exploration/holdout splitter, cross-sectional builder
src/bot/strategies/             # Strategy interface, EMA cross (test bench), Momentum-30
src/bot/risk/risk_manager.py    # position sizing and initial stop (pure function)
src/bot/backtesting/engine.py   # backtest engine, metrics, trade journal
scripts/                        # pipeline entry points (download → indicators → split → verify → backtest)
tests/                          # pytest suite
results/                        # backtest reports and trade logs (exploration only)
```

## Run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env              # API keys are not needed for public market data

python scripts/download_data.py
python scripts/add_indicators.py
python scripts/split_data.py
python scripts/verify_data.py
python scripts/backtest.py            # EMA cross on BTC 4h
python scripts/backtest_momentum.py   # Momentum-30 on BTC 1d
```

Python 3.12, pandas, numpy, ccxt, pytest. Tests run on GitHub Actions on every push.

---

Author: Ivan Biloshytskyi · [github.com/01IVAN10](https://github.com/01IVAN10)
