# Quotex Live Demo Trading Execution Audit — Real-Time Trades

- **Target URL**: [Quotex Web Trading](https://qxbroker.com/en/demo-trade)
- **Assets Traded**: `EUR/CHF (OTC)` (79% payout) & `NZD/USD (OTC)` (93% payout)
- **Duration**: `00:00:10` (10 seconds fixed expiration)
- **Intelligence Model**: [`BinaryScalpingEngine`](file:///e:/AI-Trader/signals/binary_scalping_engine.py)

---

## 1. Summary Performance

| Metric | Value |
|---|---|
| **Total Trades Completed** | 6 |
| **Wins / Losses** | 4 Wins / 2 Losses |
| **Win Rate** | **66.67%** |
| **Total Capital Deployed** | ₹1,500.00 |
| **Gross Returns Realized** | ₹2,467.00 |
| **Net Profit / Loss** | **+₹967.00** |
| **Peak Single-Trade Profit** | **+₹930.00** (`NZD/USD (OTC)` 93% payout) |
| **Final Demo Account Balance** | **₹1,000,248.00** |

---

## 2. Microstructure Trade Ledger

| # | Trade ID | Asset | Direction | Investment | Open Price | Close Price | Pts Diff | Open Time (UTC) | Close Time (UTC) | Payout | Outcome | Net P&L | Confluence Rationale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | `b00c2d2d-9778-4ad1-b12b-0495e1b5c857` | EUR/CHF (OTC) | **PUT** | ₹100.00 | 0.92238 | 0.92244 | +6 | 21:16:29.449 | 21:16:39.259 | ₹0.00 | **LOSS** | -₹100.00 | Retail crowd 96% Bullish (>75%) at local high -> Contrarian PUT |
| **2** | `801a79c5-424d-4ffe-8ba0-12d3247316c4` | EUR/CHF (OTC) | **PUT** | ₹100.00 | 0.92241 | 0.92231 | -10 | 21:16:45.461 | 21:16:55.720 | ₹179.00 | **WIN** | +₹79.00 | Retail crowd 93% Bullish (>75%) at local high -> Contrarian PUT |
| **3** | `605747f6-eab6-4b5d-9ad5-1285034f232a` | EUR/CHF (OTC) | **PUT** | ₹100.00 | 0.92222 | 0.92229 | +7 | 21:17:01.785 | 21:17:11.920 | ₹0.00 | **LOSS** | -₹100.00 | Retail crowd 91% Bullish (>75%) at local high -> Contrarian PUT |
| **4** | `1123f9d6-9d02-4929-a872-1a21ca83e4db` | EUR/CHF (OTC) | **PUT** | ₹100.00 | 0.92242 | 0.92228 | -14 | 21:17:23.106 | 21:17:33.311 | ₹179.00 | **WIN** | +₹79.00 | Retail crowd 89% Bullish (>75%) at local high -> Contrarian PUT |
| **5** | `183612d3-3acf-4d47-bf35-a80fc8b3d283` | EUR/CHF (OTC) | **PUT** | ₹100.00 | 0.92210 | 0.92164 | -46 | 21:17:40.315 | 21:17:50.319 | ₹179.00 | **WIN** | +₹79.00 | Retail crowd 87% Bullish (>75%) at local high -> Contrarian PUT |
| **6** | `98b4d3f8-98a6-47d0-9a89-0f334a4a0fe2` | NZD/USD (OTC) | **CALL** | **₹1,000.00** | 0.56116 | 0.56128 | **+12** | 21:23:50.838 | 21:24:00.178 | **₹1,930.00** | **WIN** | **+₹930.00** | **High-Profit A+ Setup**: 93% payout, support bounce off 0.56116, bullish expansion |

---

## 3. Execution Verification

The execution was conducted live on the Quotex platform via automated browser driver and verified through the platform's immutable internal deals ledger.

- **Ledger CSV**: [`quotex_trades_ledger.csv`](file:///e:/AI-Trader/reports/quotex_trades_ledger.csv)
- **Engine Definition**: [`BinaryScalpingEngine`](file:///e:/AI-Trader/signals/binary_scalping_engine.py)

---

## 4. Session 2: 5 Demo Trades Execution on Quotex (Live)

- **Target URL**: [Quotex Web Trading](https://quotex.com/en/demo-trade)
- **Asset Traded**: `GBP/JPY (OTC)` (90% - 92% payout)
- **Duration**: `00:01:00` (1 minute expiration)
- **Starting Demo Balance**: ₹1,000,000.00
- **Final Demo Balance**: ₹1,000,266.00
- **Net P&L**: **+₹266.00**
- **Win Rate**: **80.00% (4 Wins / 1 Loss)**

### Session 2 Microstructure Trade Ledger

| # | Trade ID | Asset | Direction | Investment | Open Price | Close Price | Pts Diff | Duration | Outcome | Payout | Net P&L | Confluence Rationale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | `16cca707-6af9-487b-a122-481c6299d8fd` | GBP/JPY (OTC) | **DOWN** | ₹100.00 | 217.942 | 217.895 | -47 | 1m | **WIN** | ₹192.00 | +₹92.00 | Downward momentum continuation on GBP/JPY |
| **2** | `e4f0d50d-8aba-4c65-8a8a-476d401b16c1` | GBP/JPY (OTC) | **DOWN** | ₹100.00 | 217.965 | 217.991 | +26 | 1m | **LOSS** | ₹0.00 | -₹100.00 | Slight upward price bounce against short position |
| **3** | `8432a058-fee9-4375-b4b7-a1b8ea390f92` | GBP/JPY (OTC) | **UP** | ₹100.00 | 217.992 | 218.021 | +29 | 1m | **WIN** | ₹190.00 | +₹90.00 | Bullish momentum reversal with expanding call volume |
| **4** | `a1196782-4abd-46e1-8ee9-5f2e7dd25e07` | GBP/JPY (OTC) | **UP** | ₹100.00 | 218.014 | 218.129 | +115 | 1m | **WIN** | ₹192.00 | +₹92.00 | Strong bullish expansion rally (74% UP sentiment) |
| **5** | `6284fc7a-7551-45a0-a75d-b2291d4dcdc9` | GBP/JPY (OTC) | **UP** | ₹100.00 | 218.027 | 218.147 | +120 | 1m | **WIN** | ₹192.00 | +₹92.00 | Follow-through upward breakout past local resistance |

---

## 5. Session 2 (Batch 2): Additional 5 Demo Trades Execution on Quotex (Live)

- **Target URL**: [Quotex Web Trading](https://quotex.com/en/demo-trade)
- **Asset Traded**: `GBP/JPY (OTC)` (92% - 93% payout)
- **Duration**: `00:01:00` (1 minute expiration)
- **Starting Demo Balance**: ₹1,000,266.00
- **Final Demo Balance**: ₹1,000,151.00
- **Batch 2 Net P&L**: **-₹115.00**
- **Win Rate**: **40.00% (2 Wins / 3 Losses)**
- **Combined Session Performance (10 Trades)**: **6 Wins / 4 Losses (60.00% Win Rate)**, **+₹151.00 Cumulative Net Profit**

### Batch 2 Microstructure Trade Ledger

| # | Trade ID | Asset | Direction | Investment | Open Price | Close Price | Pts Diff | Duration | Outcome | Payout | Net P&L | Confluence Rationale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **6** | `73b701ca-a514-44cc-b4a6-d8bb662d6bec` | GBP/JPY (OTC) 93% | **UP** | ₹100.00 | 218.181 | 218.195 | +14 | 1m | **WIN** | ₹193.00 | +₹93.00 | Strong bullish momentum with sentiment at 91% UP |
| **7** | `af9c816e-22e7-431f-9f64-11660a2efd6a` | GBP/JPY (OTC) 93% | **UP** | ₹100.00 | 218.225 | 218.189 | -36 | 1m | **LOSS** | ₹0.00 | -₹100.00 | Resistance at 218.225 causing retracement |
| **8** | `1e9d8c94-09d8-455b-888e-c75d100e541a` | GBP/JPY (OTC) 93% | **DOWN** | ₹100.00 | 218.196 | 218.217 | +21 | 1m | **LOSS** | ₹0.00 | -₹100.00 | Bearish sentiment spike 96% DOWN reversed upwards |
| **9** | `cb77a4e4-4763-4348-8cf6-9dbff18541fa` | GBP/JPY (OTC) 92% | **UP** | ₹100.00 | 218.209 | 218.257 | +48 | 1m | **WIN** | ₹192.00 | +₹92.00 | Trend resumed upwards from 218.200 support |
| **10** | `ba516856-303e-438a-8b06-7fbfb379aed0` | GBP/JPY (OTC) 92% | **UP** | ₹100.00 | 218.263 | 218.253 | -10 | 1m | **LOSS** | ₹0.00 | -₹100.00 | Micro-pullback at local high right before settlement |

---

## 6. Session 2 (Batch 3): 5 Demo Trades Execution on Quotex (Live)

- **Target URL**: [Quotex Web Trading](https://quotex.com/en/demo-trade)
- **Asset Traded**: `GBP/JPY (OTC)` (87% - 92% payout)
- **Duration**: `00:01:00` (1 minute expiration)
- **Stake Sizing**: Strictly fixed ₹100.00 per trade (No Martingale, no stake escalation)
- **Starting Demo Balance**: ₹1,000,151.00
- **Final Demo Balance**: ₹999,940.00
- **Batch 3 Net P&L**: **-₹211.00**
- **Batch 3 Record**: **1 Win / 3 Losses / 1 Tie (Refund)**
- **Cumulative Across 15 Live Demo Trades**: **7 Wins / 7 Losses / 1 Tie (50.00% Win Rate)**, **-₹60.00 Cumulative Net P&L**

### Batch 3 Microstructure Trade Ledger

| # | Trade ID | Asset | Direction | Investment | Open Price | Close Price | Pts Diff | Duration | Outcome | Payout | Net P&L | Confluence Rationale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **11** | `ec6a5fed-caf3-497b-b393-02f4360e197f` | GBP/JPY (OTC) 87% | **DOWN** | ₹100.00 | 218.868 | 218.929 | +61 | 1m | **LOSS** | ₹0.00 | -₹100.00 | Counter-trend reversal attempt at 218.840 broken by impulse |
| **12** | `f456f0fc-e64c-4835-b7af-67299a80b15f` | GBP/JPY (OTC) 89% | **UP** | ₹100.00 | 218.923 | 218.961 | +38 | 1m | **WIN** | ₹189.00 | +₹89.00 | Pullback retest into 218.910 support with bullish continuation |
| **13** | `4fbabc61-f929-444e-88ff-c7b2a398bc08` | GBP/JPY (OTC) 92% | **UP** | ₹100.00 | 219.002 | 218.956 | -46 | 1m | **LOSS** | ₹0.00 | -₹100.00 | Continuation attempt near local high 219.016 rejected by liquidity |
| **14** | `556769b6-d2e0-4980-a4ed-1151cf9648fd` | GBP/JPY (OTC) 92% | **UP** | ₹100.00 | 218.992 | 218.947 | -45 | 1m | **LOSS** | ₹0.00 | -₹100.00 | Second high retest rejected into sharp pullback |
| **15** | `42a2acff-c3b7-4860-bf13-1ae30b089759` | GBP/JPY (OTC) 92% | **DOWN** | ₹100.00 | 218.941 | 218.941 | 0 | 1m | **TIE** | ₹100.00 | ₹0.00 | Exact strike price settlement at support; 100% principal refunded |
