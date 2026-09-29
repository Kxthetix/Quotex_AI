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
SELECT create_hypertable('market_data.raw_market_events', 'receive_time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_raw_events_symbol_time ON market_data.raw_market_events (symbol, event_time DESC);

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
SELECT create_hypertable('market_data.candles', 'start_time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_candles_lookup ON market_data.candles (symbol, timeframe, start_time DESC);

-- 3. Normalized Trades (Tick-level clean storage)
CREATE TABLE IF NOT EXISTS market_data.normalized_trades (
    id BIGSERIAL,
    trade_id VARCHAR(64) NOT NULL,
    source VARCHAR(64) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    quantity NUMERIC(24, 8) NOT NULL,
    side VARCHAR(16) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    receive_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);
SELECT create_hypertable('market_data.normalized_trades', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON market_data.normalized_trades (symbol, timestamp DESC);

-- 4. Data Quality Incidents
CREATE TABLE IF NOT EXISTS market_data.data_quality_incidents (
    id BIGSERIAL PRIMARY KEY,
    check_name VARCHAR(64) NOT NULL,
    category VARCHAR(64) NOT NULL,
    severity VARCHAR(16) NOT NULL,
    symbol VARCHAR(32),
    message TEXT NOT NULL,
    details JSONB,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_incidents_category_time ON market_data.data_quality_incidents (category, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_symbol ON market_data.data_quality_incidents (symbol);

-- 5. Candle Corrections (Audit trail for out-of-order adjustments)
CREATE TABLE IF NOT EXISTS market_data.candle_corrections (
    correction_id VARCHAR(64) PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    original_open NUMERIC(20, 8) NOT NULL,
    original_high NUMERIC(20, 8) NOT NULL,
    original_low NUMERIC(20, 8) NOT NULL,
    original_close NUMERIC(20, 8) NOT NULL,
    original_volume NUMERIC(24, 8) NOT NULL,
    original_trades INTEGER NOT NULL,
    revised_high NUMERIC(20, 8) NOT NULL,
    revised_low NUMERIC(20, 8) NOT NULL,
    revised_close NUMERIC(20, 8) NOT NULL,
    revised_volume NUMERIC(24, 8) NOT NULL,
    revised_trades INTEGER NOT NULL,
    late_trade_id VARCHAR(64) NOT NULL,
    late_trade_price NUMERIC(20, 8) NOT NULL,
    late_trade_quantity NUMERIC(24, 8) NOT NULL,
    late_trade_timestamp TIMESTAMPTZ NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_corrections_symbol_time ON market_data.candle_corrections (symbol, start_time DESC);

-- 6. Ingestion Runs (Batch & Backfill execution ledger)
CREATE TABLE IF NOT EXISTS market_data.ingestion_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    provider VARCHAR(64) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL DEFAULT '1m',
    requested_start TIMESTAMPTZ,
    requested_end TIMESTAMPTZ,
    actual_start TIMESTAMPTZ,
    actual_end TIMESTAMPTZ,
    status VARCHAR(32) NOT NULL,
    records_received INTEGER NOT NULL DEFAULT 0,
    records_accepted INTEGER NOT NULL DEFAULT 0,
    records_rejected INTEGER NOT NULL DEFAULT 0,
    duplicates INTEGER NOT NULL DEFAULT 0,
    errors JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completion_time TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_runs_provider_symbol ON market_data.ingestion_runs (provider, symbol, created_at DESC);

-- 7. Ingestion Health & Metrics Log
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
