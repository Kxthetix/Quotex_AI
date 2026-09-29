# Phase 5 — Execution and Market Microstructure Model

## 1. Explicit Order Representation

Orders (`backtesting/models.py`) are first-class domain objects with explicit lifecycle tracking:

- **Order Types:** `MARKET`, `LIMIT`, `STOP`.
- **Order Sides:** `BUY`, `SELL`.
- **Order States:** `CREATED` -> `SUBMITTED` -> `PARTIALLY_FILLED` -> `FILLED`, `CANCELLED`, `REJECTED`, `EXPIRED`.
- **Order Properties:** `order_id`, `symbol`, `quantity`, `filled_qty`, `remaining_qty`, `limit_price`, `stop_price`, `submitted_at`, `executed_at`, `cumulative_fees`, `cumulative_slippage`.

## 2. Realistic Execution Simulator

The `ExecutionSimulator` (`backtesting/execution/simulator.py`) models real-world exchange frictions:

### A. Bid/Ask Spread Model
Derives synthetic bid and ask around the reference candle price:
- **Buy orders** execute at the Ask (higher price).
- **Sell orders** execute at the Bid (lower price).
- Half-spread cost is explicitly attributed to every fill.

### B. Slippage Model
Deterministic adverse price movement:
- **Buy orders** slip upward ($P_{fill} = P_{ask} \times (1 + slippage)$).
- **Sell orders** slip downward ($P_{fill} = P_{bid} \times (1 - slippage)$).
- Supports volatility-scaled slippage multipliers.

### C. Execution Latency Model
Simulates physical delay between order submission and exchange matching:
$$T_{eligible} = T_{submitted} + \Delta t_{latency}$$
Orders cannot execute until market candles reach $T \ge T_{eligible}$. Execution price is derived from market data available at $T_{eligible}$, never at $T_{signal}$.

### D. Liquidity and Partial Fills
- Limits individual fill size to a configured fraction of the candle's total volume (`max_bar_volume_pct`, default 10%).
- If order quantity exceeds bar capacity, a partial fill is executed, and remaining quantity remains active for subsequent bars.

### E. Transaction Fee Model
- Clearly distinguishes Maker fees, Taker fees, and fixed ticket fees.
- Fees are deducted from cash balances and attributed per trade.
