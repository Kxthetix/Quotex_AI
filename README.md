# Trading Intelligence System

A serious, local market-analysis and algorithmic trading platform designed for continuous market ingestion, multi-timeframe feature analysis, statistical and machine-learning signal fusion, and a controlled, verifiable self-learning loop.

> **Core Objective**: Produce high-confidence `BUY` / `SELL` / `NO-TRADE` decisions, record every prediction to an immutable ledger, rigorously evaluate outcomes against actual market movements, and systematically improve through an audited, non-mutating validation and governance loop.

---

## High-Level System Architecture

```
                      EXCHANGE / MARKET SOURCES
                                 │
                                 ▼
                     DATA ACQUISITION (Live WS / REST)
                                 │
                                 ▼
                        DATA QUALITY ENGINE
               (Timestamp, OHLC, Outlier & Stale Checks)
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
        HISTORICAL STORE                  LIVE MARKET STATE
    (TimescaleDB / Parquet)                (Redis / Memory)
                │                                 │
                └────────────────┬────────────────┘
                                 ▼
                           CANDLE BUILDER
                                 │
                                 ▼
                           FEATURE ENGINE  [PHASE 3]
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
        TECHNICAL            HISTORICAL         ML / STATISTICAL
         ANALYSIS             PATTERNS             PREDICTION
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 ▼
                           MARKET REGIME
                                 │
                                 ▼
                           SIGNAL FUSION
                                 │
                       BUY / SELL / NO-TRADE
                                 │
                                 ▼
                            RISK ENGINE
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
           PAPER TRADING                     EXECUTION
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                          OUTCOME TRACKER
                                 │
                                 ▼
                           MODEL AUDITOR
                                 │
                                 ▼
                          LEARNING ENGINE
                                 │
                                 ▼
                          CANDIDATE MODEL
                                 │
                                 ▼
                     BACKTEST & WALK-FORWARD
                                 │
                                 ▼
                     OUT-OF-SAMPLE VALIDATION
                                 │
                                 ▼
                     PAPER-TRADING VALIDATION
                                 │
                                 ▼
                         MODEL GOVERNANCE
                         (Promote / Reject)
                                 │
                                 ▼
                         PRODUCTION MODEL
                                 │
                                 └───────────────> (Repeat)
```

---

## Phase 2: Verified Data Pipeline

Phase 2 establishes a complete, provider-neutral market data layer capable of handling both live WebSocket streaming and historical backfill data with zero schema divergence.

```
                    MARKET SOURCE (Binance WS / REST)
                                   │
                                   ▼
                    DATA ACQUISITION (Connector)
                                   │
                                   ▼
                    EVENT VALIDATOR (DataQualityEngine)
                                   │
                                   ▼
                       NORMALIZER (EventNormalizer)
                                   │
             ┌─────────────────────┴─────────────────────┐
             ▼                                           ▼
       RAW DATA STORE                              MARKET STATE
  (Immutable JSONL / DB)                         (Redis / Memory)
                                                         │
                                                         ▼
                                                   CANDLE BUILDER
                                              (1m OHLCV Aggregator)
                                                         │
                                                         ▼
                                                  CLEAN DATA STORE
                                               (Parquet / TimescaleDB)
```

### Core Phase 2 Capabilities

1. **Provider-Neutral Abstraction**:
   - `MarketDataInterface` isolates the system from specific exchange protocols.
   - Concrete `BinanceWebSocketConnector` implements both `WebSocketMarketDataProvider` and `HistoricalMarketDataProvider`.
   - Exchange payload quirks are strictly isolated inside the connector; domain logic receives only canonical models.

2. **Unified Data Normalization**:
   - **Mandatory Principle**: Live data and historical backfills produce the exact same canonical schemas (`NormalizedTrade` and `Candle`).
   - Timestamps are normalized to timezone-aware UTC datetime.
   - Symbols are uppercase normalized.
   - Price and volume values are validated as IEEE floating-point numerics.

3. **Immutable Raw Data Storage**:
   - Every raw event received is persisted immutably in `data/storage/raw/` (`raw_events_*.jsonl` and `raw_hist_*.jsonl`).
   - Captures provider/source, symbol, event type, exchange timestamp (`event_time`), local ingestion timestamp (`receive_time`), and the unmodified raw payload dictionary.
   - Raw source records are never modified or cleaned in place.

4. **Structured Event Validation (8 Required Categories)**:
   - Evaluated by `DataQualityEngine`:
     - `INVALID_TIMESTAMP`: Clock drift beyond tolerance (>5s into future) or invalid time bounds.
     - `MISSING_FIELD`: Missing mandatory symbol, price, volume, or event type.
     - `INVALID_PRICE`: Non-positive (<=0) or non-numeric prices.
     - `INVALID_VOLUME`: Non-positive (<=0) or non-numeric quantities/volumes.
     - `UNKNOWN_EVENT_TYPE`: Unrecognized or malformed stream action.
     - `MALFORMED_PAYLOAD`: Non-dictionary or corrupted frames.
     - `DUPLICATE_EVENT`: Repeated trade IDs / message duplicates.
     - `OUT_OF_ORDER_EVENT`: Ticks arriving chronologically older than previously observed events.
     - `PRICE_OUTLIER`: Price jump exceeding threshold (default 20%).
     - `CANDLE_OHLC_INCONSISTENCY`: High < Open, High < Close, Low > Open, or Low > Close.
     - `MISSING_INTERVAL`: Sequence gaps in expected candle continuity.
     - `STALE_DATA`: Feed inactivity exceeding timeout (default 15s).

5. **Auditable Late-Event Correction Policy**:
   - In live trading, network jitter can cause trades to arrive out of order or after a 1-minute boundary has closed.
   - If a late trade arrives for an already-closed candle within the tolerance window (default 5 minutes), the system **never silently mutates** history.
   - Instead, it generates an immutable `CandleCorrectionRecord` detailing:
     - `correction_id`, `symbol`, `timeframe`, `start_time`
     - `original_open`, `original_high`, `original_low`, `original_close`, `original_volume`, `original_trades`
     - `revised_high`, `revised_low`, `revised_close`, `revised_volume`, `revised_trades`
     - `late_trade_id`, `late_trade_price`, `late_trade_quantity`, `late_trade_timestamp`
   - Dispatched to persistence and logged with warning severity.

6. **Missing Candle & Sequence Gap Detection**:
   - `MissingDataDetector` analyzes candle series and classifies gaps:
     - `PROVIDER_DATA_DROPOUT`: Data feed dropped during active 24/7 hours.
     - `MARKET_SESSION_CLOSED`: Legitimate closure (e.g. weekend spanning for traditional equities/forex).
     - `NETWORK_OUTAGE`: Correlated with connection manager dropouts.
     - `NO_TRADE_PERIOD`: Isolated 1-2 minute period with zero trade volume.

7. **Historical Ingestion Idempotency & Run Tracking**:
   - Historical imports are tracked via the `IngestionRun` ledger (`run_id`, requested/actual start & end, records received, accepted, rejected, duplicates, errors).
   - Re-importing the same provider, symbol, timeframe, and date range detects the completed run and skips duplicate work.

8. **Asynchronous Non-Blocking Persistence**:
   - Processing flow: `receive -> validate -> normalize -> in-memory state -> candle build -> async persistence enqueue`.
   - `AsyncPersistenceWorker` batches raw events, candles, normalized trades, quality incidents, and candle corrections without blocking the low-latency processing loop.

---

## Repository Structure

```
AI-Trader/
├── data/
│   ├── ingestion/             # WebSocket connector, ConnectionManager, CandleBuilder, IngestionPipeline, HistoricalDataImporter
│   ├── validation/            # DataQualityEngine, MissingDataDetector, ValidationCategory models
│   ├── normalization/         # EventNormalizer mapping exchange payloads to canonical models
│   └── storage/               # PostgreSQL/TimescaleDB models, Parquet store, Redis state, IngestionRunRepository
├── features/                  # [Phase 3] Technical, volume, momentum, volatility indicators
├── analysis/                  # [Phase 4] Technical regimes, historical pattern matching
├── models/                    # [Phase 5] Direction, return, volatility ML & ensemble models
├── signals/                   # [Phase 6] Signal generation, confidence calibration, fusion
├── risk/                      # [Phase 7] Position sizing, exposure limits, drawdown breakers
├── backtesting/               # [Phase 8] Chronological event-driven replay engine
├── paper_trading/             # [Phase 9] Simulated execution, prediction ledger, outcome tracking
├── execution/                 # [Phase 10] Order router & broker execution abstraction
├── learning/                  # [Phase 11] Model auditor, candidate retraining, validation gates, governance
├── monitoring/                # Structured logging, latency percentiles, system health
├── api/                       # Operational FastAPI telemetry, health, and quality endpoints
├── configs/                   # Type-safe configuration management (Pydantic Settings)
├── docker/                    # TimescaleDB initialization scripts and container configs
├── scripts/                   # CLI runners (run_live_pipeline, import_historical, verify_live_stream)
├── tests/                     # Unit and integration test suites (Groups A through I)
├── pyproject.toml             # Python packaging, dependency specification, tool configurations
└── README.md
```

---

## Prerequisites

- **Operating System**: Windows (Runs natively via PowerShell; no WSL, Linux VMs, or cloud infrastructure needed for V1)
- **Python**: Python 3.12+ (tested and verified with Python 3.13)
- **Git**: Installed and available on system PATH
- **Docker Desktop**: For running local TimescaleDB, Redis, and MLflow services

---

## Local Setup Instructions (Windows)

### 1. Clone & Enter Directory
```powershell
cd E:\AI-Trader
```

### 2. Create Virtual Environment
```powershell
py -3.13 -m venv .venv
```

### 3. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env` (pre-configured with safe defaults and zero hardcoded secrets):
```powershell
Copy-Item .env.example .env
```

### 6. Launch Local Infrastructure (Docker Desktop)
Launch TimescaleDB, Redis, and MLflow:
```powershell
docker compose up -d
```

### 7. Run Database Migrations
Initialize database tables, hypertables, and indexes:
```powershell
alembic upgrade head
```

---

## Execution Guide

### Verify Live Market Data Connectivity
Run the quick verification script to connect to Binance's public WebSocket stream and inspect live incoming ticks:
```powershell
python scripts/verify_live_stream.py
```

### Start the Live Ingestion Pipeline
Start the continuous live streaming pipeline with real-time 1-minute candle building, latency telemetry, and circuit breaker tracking:
```powershell
python scripts/run_live_pipeline.py
```

### Start the Telemetry & Health API
Launch the FastAPI telemetry server to monitor connection health, latency percentiles, market states, and data quality metrics:
```powershell
uvicorn api.main:app --reload --port 8000
```
- Interactive API Docs: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/health`
- Live Market State: `http://localhost:8000/api/v1/market/state/BTCUSDT`
- Data Quality Statistics: `http://localhost:8000/api/v1/market/quality/stats`
- Quality Incidents Ledger: `http://localhost:8000/api/v1/market/quality/incidents`
- Circuit Breaker Reset: `POST http://localhost:8000/api/v1/market/breaker/reset`

### Import Historical Market Data
Download and validate historical 1-minute candles from the REST API directly into structured Parquet and TimescaleDB storage with audit run tracking:
```powershell
# Standard historical import (last 500 1-minute candles)
python scripts/import_historical.py --symbol BTCUSDT --timeframe 1m --limit 500

# Explicit date range import
python scripts/import_historical.py --symbol BTCUSDT --timeframe 1m --start 2026-01-01T00:00:00 --end 2026-01-02T00:00:00

# Force re-import bypassing idempotency check
python scripts/import_historical.py --symbol BTCUSDT --timeframe 1m --limit 500 --force
```

### Calculate & Inspect Versioned Features (Phase 3)
Compute versioned, deterministic, leakage-safe feature vectors for historical data or synthetic streams:
```powershell
# Calculate features on 300 candles and persist to Parquet
python scripts/calculate_features.py --symbol BTC/USDT --count 300 --output data/features/btc_v1.parquet

# Calculate features from an imported candle Parquet file
python scripts/calculate_features.py --symbol BTCUSDT --input-parquet data/storage/parquet/binance/BTCUSDT/1m/2026-01.parquet --output data/features/btc_historical_v1.parquet
```

---

## Phase 3 — Feature Engine

The Feature Engine transforms normalized market data from Phase 2 into version-locked, deterministic, and leakage-safe `FeatureVector` collections.

### Core Architecture
```
                 NORMALIZED MARKET DATA (Candles)
                               │
                               ▼
                        FEATURE ENGINE
                               │
         ┌─────────────┬───────┴─────┬────────────┬─────────────┐
         ▼             ▼             ▼            ▼             ▼
       PRICE         TREND       MOMENTUM    VOLATILITY       VOLUME
     (Returns,     (EMA, SMA,   (RSI, MACD,   (ATR, Realized  (Relative Vol,
     Wicks, Body,   ADX, Slope,   ROC, Stoch)  Vol, Hist Vol)  Spikes, Trend)
     Range, Gap)    Structure)
         │             │             │            │             │
         └─────────────┼─────────────┼────────────┴─────────────┘
                       ▼
                   STRUCTURE (Support, Resistance, Breakout, Consolidation)
                       │
                       ▼
            MULTI-TIMEFRAME ENGINE (5m, 15m, 1h, 4h, 1d)
                       │
                       ▼
             VERSIONED FEATURE VECTOR (v1.0.0)
                       │
                       ▼
             VERSIONED PERSISTENCE (Parquet / TimescaleDB)
```

### Key Invariants
1. **Separation of Concerns**: A feature strictly answers *"What was measurable from market data at this timestamp?"* Features **never** answer *"Should I buy or sell?"*
2. **Strict Anti-Leakage (Core Invariant)**: For every feature, $\text{FEATURE}(\text{symbol}, \text{timeframe}, T)$ depends solely on market observations available at or before $T$.
3. **Multi-Timeframe Alignment**: At timestamp $T$, higher-timeframe features (5m, 15m, 1h, 4h, 1d) only match completed candles where `end_time <= T`. In-progress higher-timeframe candles are strictly excluded via backward merge (`pd.merge_asof(..., direction='backward')`).
4. **Transparent Warm-Up Behavior**: During insufficient history, features evaluate to `None` / `NaN` with `FeatureValidity.WARMING_UP`. Features are **never** silently replaced with `0.0`.
5. **Exact Historical/Live Consistency**: Batch historical calculation and incremental rolling buffer live calculation produce 100% identical outputs on identical completed bars.

For full formulas, lookbacks, and parameters for all 92 features, see the [Feature Catalog](docs/feature_catalog.md).

---

## Phase 4 — Baseline Intelligence

Phase 4 establishes the scientifically testable baseline intelligence layer. It answers:
> *"Given only information available at time $T$, what does the system's historical evidence and baseline models say about what may happen next?"*

> **Mandatory Invariant**: Phase 4 does **NOT** decide *"Should the system place a trade?"* and outputs **NO** production trade execution instructions (`BUY`/`SELL`). Signal fusion and risk execution are reserved for Phase 6 and 7.

### Architecture

```
                 FEATURE ENGINE (Phase 3)
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   TECHNICAL RULES    REGIME ENGINE    PATTERN ENGINE
       (v1.0.0)          (v1.0.0)      (Vector & Sequence)
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                      LABEL ENGINE
         (direction, return, volatility, trade-quality)
                            │
                            ▼
                    DATASET GENERATOR
                 (Chronological Splitter)
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
     STATISTICAL        SPECIALIZED       BASE REGIME
       MODELS            ML MODELS           MODEL
     (Majority,         (Direction,       (Learned RF/
      Mean, Reg)      Return, Vol, TQ)       LGBM)
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                  CONFIDENCE CALIBRATION
                     (Platt / Isotonic)
                            │
                            ▼
                   EVALUATION INTERFACES
              (Classification / Regression / ECE)
                            │
                            ▼
              LOCAL MODEL REGISTRY & MLFLOW
```

### CLI Quickstart Commands (Phase 4 — Baseline Intelligence)

#### 1. Generate Version-Locked Dataset
```powershell
python scripts/generate_dataset.py --symbol BTC/USDT --candles 500 --output data/datasets/btc_1m.parquet
```

#### 2. Run Historical Pattern & Sequence Matching
```powershell
python scripts/run_pattern_search.py --top-k 5 --sequence-window 20
```

#### 3. Train, Calibrate, and Register Baseline Models
```powershell
# Train Direction Classifier (Logistic Regression, Platt Calibration, MLflow tracking)
python scripts/train_baseline.py --model direction --algo logistic_regression

# Train Return Magnitude Regressor (Ridge Regression)
python scripts/train_baseline.py --model return --algo ridge

# Train Statistical Baselines
python scripts/train_baseline.py --model majority_class
python scripts/train_baseline.py --model historical_mean
```

#### 4. Launch MLflow UI to Inspect Experiments
```powershell
mlflow ui --backend-store-uri file:///E:/AI-Trader/data/mlruns
```

---

## Phase 5: Verified Backtesting & Walk-Forward Engine

Phase 5 establishes an event-driven chronological simulation environment replicating real-world market microstructure:

- **Historical Replay Engine:** Chronological streaming enforcing that a candle completed at $T$ is only observable at $\ge T$.
- **Realistic Execution Simulation:** Explicit order models (`MARKET`, `LIMIT`, `STOP`) with configurable bid/ask spread, adverse slippage, latency delays, maker/taker fees, and liquidity-capped partial fills.
- **Position & Portfolio Accounting:** Dynamic asset-liability cash accounting, gross/net exposures, time-series equity curve snapshots, and immutable trade records.
- **Quantitative Performance Engine:** Total return, CAGR, continuous max drawdown and duration, annualized Sharpe, Sortino, Calmar, win rate, profit factor, expectancy, full cost attribution, and regime breakdowns.
- **Rolling Walk-Forward Cross-Validation:** Train/Validation/Test window rolling evaluations enforcing strict anti-leakage invariants ($\max(T_{train}) < \min(T_{val}) < \min(T_{test})$).

### CLI Quickstart Commands (Phase 5 — Backtesting Engine)

#### 1. Run Chronological Backtest with Realistic Microstructure Friction
```powershell
python scripts/run_backtest.py --data-file data/storage/parquet/BTCUSDT/BTCUSDT_1m_20260928.parquet --initial-capital 10000 --spread-bps 2.0 --slippage-bps 3.0 --latency-sec 1.0
```
Use `--synthetic-smoke-test --candles 600` only to inspect the runner; generated-candle results are not evidence of strategy performance.

#### 2. Run Rolling Walk-Forward Cross-Validation
```powershell
python scripts/run_walk_forward.py --data-file data/storage/parquet/BTCUSDT/BTCUSDT_1m_20260928.parquet --train-bars 400 --val-bars 100 --test-bars 100 --step-bars 100
```

#### 3. Inspect Generated Reports and Ledgers
- Backtest Markdown Report: `reports/backtest_report.md`
- Portfolio Equity Curve CSV: `reports/equity_curve.csv`
- Completed Trade Ledger CSV: `reports/trade_ledger.csv`
- Walk-Forward Validation Report: `reports/walk_forward_report.md`

---

## Testing & Quality Assurance

### Run Full Test Suite
Executes all 160 unit and integration tests across Phase 1, Phase 2, Phase 3, Phase 4, and Phase 5:
```powershell
pytest -v
```

### Run Ruff Linter & Formatter Check
```powershell
ruff check .
ruff format --check .
```

---

## Data Differences & Assumptions

### Raw vs. Clean Data
| Attribute | Raw Storage Layer | Clean Storage Layer |
|---|---|---|
| **Location** | `data/storage/raw/*.jsonl` & `raw_market_events` | `data/storage/parquet/` & `market_candles` |
| **Immutability** | 100% Immutable write-only append | Audited updates via `CandleCorrectionRecord` |
| **Payload** | Exact exchange dictionary structure | Canonical Pydantic model (`NormalizedTrade` / `Candle`) |
| **Timestamps** | Exchange `event_time` & local `receive_time` | Normalized UTC datetime (`start_time`, `end_time`) |
| **Validation** | Raw payload syntax & timestamp drift | Full 8-category validation & outlier verification |

### Provider-Specific Assumptions (Binance)
1. **Public Market Stream**: Connects to `wss://stream.binance.com:9443/ws` and `https://api.binance.com`. Requires no API keys.
2. **Stream Topics**: Subscribes to `<symbol>@trade` for individual execution ticks and `<symbol>@kline_1m` for 1-minute candles.
3. **Buyer Is Maker (`m` field)**: In Binance trade payloads, `m=True` indicates the buyer was the maker (passive limit order), meaning the aggressor was a seller (`side = "sell"`). `m=False` indicates the aggressor was a buyer (`side = "buy"`).
4. **24/7 Market Semantics**: Crypto markets trade 24 hours a day, 7 days a week; sequence gaps are classified as `PROVIDER_DATA_DROPOUT` or `NETWORK_OUTAGE`, not market session closures.
5. **No Synthetic Fill Assumptions**: If a minute contains zero trades, the system does not fabricate synthetic candles. Gaps are auditable.
