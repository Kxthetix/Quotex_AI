# Multi-Provider Architecture

## 1. Provider Abstraction

Market data feeds implement the canonical `MarketDataProvider` contract:
- `connect()`, `disconnect()`
- `is_connected`: Real-time socket and session status.
- `subscribe_trades(symbols)`
- `subscribe_candles(symbols, timeframe)`
- `subscribe_orderbook(symbols, depth)`
- `stream_events()`: Asynchronous generator yielding canonical `MarketEvent` instances.
- `fetch_historical_candles()`
- `fetch_orderbook_snapshot()`
- `get_health()`: Diagnostic metrics (message count, latency, connection state).

## 2. Canonical Normalization

Regardless of whether market data originates from Binance, Coinbase, Alpaca, or synthetic simulations, incoming ticks and depth updates are normalized into canonical events. Downstream feature generators and models remain completely agnostic to provider identity.
