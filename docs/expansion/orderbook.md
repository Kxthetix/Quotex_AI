# Order-Book Architecture and Sequence Validation

## 1. Schema & Level Representation

The order book represents Level 2 market depth:
- `OrderBookLevel`: `price: float`, `size: float`, `level: int` (1-indexed).
- `OrderBookSnapshot`: Point-in-time state containing sorted bids (descending), sorted asks (ascending), best bid/ask, spread, and mid-price.
- `OrderBookUpdate`: Incremental delta update with sequence number and price/quantity adjustments.

## 2. Sequence Gap Detection & State Invariants

The `OrderBookTracker` maintains local depth from initial snapshots and incremental deltas:
- **Sequence Continuity**: For every update, $\text{seq}_{\text{new}} == \text{seq}_{\text{last}} + 1$.
- **Gap Detection**: If an update arrives out of sequence:
  1. Book status transitions to `GAP_DETECTED`.
  2. Feature generation for the affected instrument is immediately HALTED.
  3. Snapshot rebuild request is issued.
- **Book Invariants**:
  - Uncrossed book: $\text{best\_bid} < \text{best\_ask}$. If best bid $\ge$ best ask, status transitions to `CROSSED_BOOK`.
  - Non-negative prices and positive quantities.
