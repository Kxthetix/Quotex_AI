# Paper Execution & Portfolio Accounting

## 1. Safety Enforcements

The `PaperExecutionEngine` implements strict checks preventing real execution:
- Rejects real broker connections, exchange keys, and production endpoints.
- Verifies all orders remain purely internal `PaperOrder` instances.
- Raises `RealExecutionBlockedError` if an external broker integration is configured.

## 2. Realistic Execution Frictions

Simulated execution avoids unrealistic "instant zero-cost fills":

1. **Spread Simulation**:
   - Uses live bid/ask quotes when available ($P_{\text{buy}} = P_{\text{ask}}$, $P_{\text{sell}} = P_{\text{bid}}$).
   - Defaults to configured spread percentage model (e.g., 2 to 4 bps).

2. **Slippage Model**:
   - Accounts for base slippage plus market impact:
     $$\text{Slippage} = \text{Base} + \left(\frac{\text{Order Qty}}{\text{Bar Volume}}\right) \times \text{VolumeFactor} + (\sigma \times 0.05)$$
   - Slippage moves against the trader on both buys and sells.

3. **Fees**:
   - Differentiates Maker fees (default: 0.02%) from Taker fees (default: 0.05%).
   - Limit orders executing within the spread incur maker fees; market orders incur taker fees.

4. **Simulated Latency**:
   - Generates realistic execution delays (20ms to 120ms).
   - Informs latency metrics ($p50, p95, p99$).

5. **Partial Fills**:
   - Enforces bar volume capacity limits (`max_bar_volume_pct`: default 20%).
   - Orders exceeding available volume generate partial fills across successive events.

## 3. Paper Portfolio & Trade Ledger

- **Cash Accounting**:
  - Buying deducts cash: $\text{Cash} \leftarrow \text{Cash} - (\text{Price} \times \text{Qty} + \text{Fees})$.
  - Selling adds cash: $\text{Cash} \leftarrow \text{Cash} + (\text{Price} \times \text{Qty} - \text{Fees})$.
- **Equity Valuation**:
  $$\text{Total Equity} = \text{Cash} + \text{Total Market Value}$$
- **Drawdown**:
  $$\text{Drawdown} = \frac{\text{Peak Equity} - \text{Current Equity}}{\text{Peak Equity}}$$
- **Trade Ledger**:
  - Automatically records completed round-trip trades with entry price, exit price, fees, slippage, holding duration, and exit reason.
