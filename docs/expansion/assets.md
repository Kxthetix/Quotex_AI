# Multi-Asset Abstraction and Symbol Mapping

## 1. Asset Model

Every tradable instrument is represented by a canonical `Asset` instance:
- `asset_id`: Globally unique identifier (e.g. `asset_btc_usdt`).
- `symbol`: Canonical human-readable symbol (e.g. `BTC/USDT`).
- `asset_class`: `CRYPTO`, `EQUITY`, `FX`, `COMMODITY`, `FUTURES`.
- `exchange`: Primary execution or market venue (e.g. `BINANCE`, `NASDAQ`).
- `quote_currency`: Settlement currency (e.g. `USDT`, `USD`).
- `base_currency`: Base asset (e.g. `BTC`, `AAPL`).
- `trading_hours`: Operational schedule (e.g. `24/7` or `09:30-16:00 EST`).
- `tick_size`: Minimum price increment (e.g. `0.01`).
- `lot_size`: Minimum order quantity increment (e.g. `0.0001`).
- `contract_specs`: Multipliers, expiry dates, or margin requirements where applicable.
- `active`: Operational availability flag.

## 2. Symbol Normalization

Different exchanges and feeds identify the same market instrument using divergent ticker symbols:

| Canonical ID | Canonical Symbol | Binance | Coinbase | Kraken | Mock Feed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `asset_btc_usdt` | `BTC/USDT` | `BTCUSDT` | `BTC-USD` | `XBT/USD` | `BTC/USDT` |
| `asset_eth_usdt` | `ETH/USDT` | `ETHUSDT` | `ETH-USD` | `ETH/USD` | `ETH/USDT` |
| `asset_sol_usdt` | `SOL/USDT` | `SOLUSDT` | `SOL-USD` | `SOL/USD` | `SOL/USDT` |

## 3. Strict Rejection of Unknown Symbols

Section 9: The system never guesses or auto-completes unknown symbols. If an incoming message contains an unregistered ticker, `AssetRegistry.normalize_provider_symbol()` raises a `KeyError` and rejects the event before feature generation.
