# Trading Intelligence System

A serious, local market-analysis and algorithmic trading platform designed for continuous market ingestion, multi-timeframe feature analysis, statistical and machine-learning signal fusion, and a controlled, verifiable self-learning loop.

> **Core Objective**: Produce high-confidence `BUY` / `SELL` / `NO-TRADE` decisions, record every prediction to an immutable ledger, rigorously evaluate outcomes against actual market movements, and systematically improve through an audited, non-mutating validation and governance loop.

---

## Architecture Overview

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
                           FEATURE ENGINE
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

## Milestone 1: Verified Live Data Pipeline

Phase 1 establishes a production-grade live market data pipeline running natively on Windows:

- **Provider-Neutral Abstraction**: `MarketDataInterface` isolates data ingestion from upstream exchanges, permitting provider replacement without changing intelligence or risk engines.
- **First Provider Connector**: `BinanceWebSocketConnector` consuming public streaming trades and REST klines without requiring paid subscriptions or API credentials.
- **Enterprise Data Quality Engine**: Rigorous validation checks for timestamp clock drift, duplicates, missing candle sequences, OHLC consistency, price outlier spikes (>15-20%), and stale data. Activates automated data-quality circuit breakers upon repeated anomalies.
- **1-Minute Candle Builder**: Aggregates raw trades into clean 1-minute OHLCV candles in memory, handling boundary transitions and late-tick discards.
- **Asynchronous Persistence**: Live events are processed in memory and queued asynchronously for storage, decoupling real-time execution from disk/database I/O.
- **Telemetry & Latency Tracking**: Computes true p50, p95, and p99 processing latencies across a sliding observation window.
- **Operational Health API**: FastAPI service exposing health reports, latency percentiles, in-memory market states, and circuit breaker controls.

---

## Repository Structure

```
trading-system/
├── data/
│   ├── ingestion/             # WebSocket connector, ConnectionManager, CandleBuilder, Pipeline
│   ├── validation/            # DataQualityEngine, validation models, circuit breakers
│   ├── normalization/         # EventNormalizer mapping exchange payloads to canonical models
│   └── storage/               # PostgreSQL/TimescaleDB models, Parquet store, Redis state, migrations
├── features/                  # Price, volume, momentum, volatility, structure, multi-timeframe
├── analysis/                  # Technical, regime, historical, pattern matching
├── models/                    # Direction, return, volatility, regime, ensemble models
├── signals/                   # Signal generation, fusion, confidence calibration
├── risk/                      # Position sizing, exposure limits, drawdown breakers
├── backtesting/               # Chronological event-driven replay engine
├── paper_trading/             # Simulated execution, prediction ledger, outcome tracking
├── execution/                 # Order router & exchange execution abstraction
├── learning/                  # Model auditor, candidate retraining, validation gates, governance
├── monitoring/                # Structured logging, latency percentiles, system health
├── api/                       # Operational FastAPI telemetry and control endpoints
├── configs/                   # Type-safe configuration management (Pydantic Settings)
├── docker/                    # TimescaleDB initialization scripts and container configs
├── scripts/                   # CLI runners (run_live_pipeline, import_historical, verify_live_stream)
├── tests/                     # Unit and integration test suites
├── pyproject.toml             # Python packaging, dependency specification, tool configurations
└── README.md
```

---

## Prerequisites

- **Operating System**: Windows (Runs natively via PowerShell; no WSL, Linux VMs, or Kubernetes required for V1)
- **Python**: Python 3.12+ (tested and verified with Python 3.13 / 3.14)
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
Copy `.env.example` to `.env` (pre-configured for local development):
```powershell
Copy-Item .env.example .env
```

### 6. Launch Local Infrastructure (Docker Desktop)
Ensure Docker Desktop is running, then launch TimescaleDB, Redis, and MLflow:
```powershell
docker compose up -d
```

### 7. Run Database Migrations
Initialize the database schema and TimescaleDB hypertables:
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
Launch the FastAPI telemetry server to monitor connection health, latency percentiles, and market states:
```powershell
uvicorn api.main:app --reload --port 8000
```
- Interactive API Docs: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/health`
- Live Market State: `http://localhost:8000/api/v1/market/state/BTCUSDT`

### Import Historical Market Data
Download and validate historical 1-minute candles from the REST API directly into structured Parquet and database storage:
```powershell
python scripts/import_historical.py --symbol BTCUSDT --timeframe 1m --limit 500
```

---

## Testing & Quality Assurance

### Run Full Test Suite
```powershell
pytest -v
```

### Run Ruff Linter & Formatter Check
```powershell
ruff check .
ruff format --check .
```

---

## Operational Safety Rules (Phase 1)

1. **Prediction Isolation**: Prediction code never places real-money orders.
2. **Circuit Breakers**: Repeated malformed events or price outlier jumps immediately trip the `DataQualityEngine` circuit breaker, halting downstream signal generation.
3. **Disconnect Rule**: Any connection disruption or stale-data timeout immediately flags the pipeline as unhealthy and pauses live decision-making.
4. **Data Immutability**: Raw market data is stored immutably; cleaned normalized data is kept strictly partitioned.
5. **No Lookahead / Leakage**: Feature and candle builders operate strictly on past and closed intervals.
