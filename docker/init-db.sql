-- Initialize TimescaleDB extension and core schemas
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS intelligence;
CREATE SCHEMA IF NOT EXISTS governance;

-- 1. Raw Market Events (Immutable audit log)
CREATE TABLE IF NOT EXISTS market_data.raw_market_events (
    id BIGSERIAL,
    source VARCHAR(64) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    receive_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload JSONB NOT NULL,
    PRIMARY KEY (id, receive_time)
);

-- Turn raw events into a TimescaleDB hypertable partitioned by receive_time
SELECT create_hypertable('market_data.raw_market_events', 'receive_time', if_not_exists => TRUE);

-- 2. Cleaned Normalized Candles
CREATE TABLE IF NOT EXISTS market_data.candles (
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(24, 8) NOT NULL,
    trades_count INTEGER DEFAULT 0,
    is_closed BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, start_time)
);

-- Turn candles into a hypertable partitioned by start_time
SELECT create_hypertable('market_data.candles', 'start_time', if_not_exists => TRUE);

-- Indexes for high-speed time-series queries
CREATE INDEX IF NOT EXISTS idx_candles_lookup ON market_data.candles (symbol, timeframe, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_raw_events_symbol_time ON market_data.raw_market_events (symbol, event_time DESC);

-- 3. Ingestion Health & Metrics Log
CREATE TABLE IF NOT EXISTS market_data.ingestion_health_log (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL,
    p50_latency_ms DOUBLE PRECISION,
    p95_latency_ms DOUBLE PRECISION,
    p99_latency_ms DOUBLE PRECISION,
    events_per_sec DOUBLE PRECISION,
    disconnections_count INTEGER DEFAULT 0,
    details JSONB,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
