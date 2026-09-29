# Expansion API Surface

## 1. Versioned Endpoints (`/api/v1`)

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/assets` | GET | List registered canonical assets and symbol mappings. |
| `/api/v1/market-data/orderbook` | GET | Retrieve latest validated Level 2 order-book snapshot. |
| `/api/v1/features/microstructure` | GET | Query real-time `orderbook_features_v1` metrics. |
| `/api/v1/models` | GET | Inspect registered models and governance statuses. |
| `/api/v1/models/predict` | POST | Request directional sequence inference. |
| `/api/v1/orders` | POST | Submit risk-approved order via ExecutionCoordinator. |
| `/api/v1/positions` | GET | Query current account positions and reconciliation state. |
| `/api/v1/reconciliation/status`| GET | Inspect position discrepancy status. |

## 2. Request Security

- API requests require token authentication and tenant scoping headers.
- Secrets are masked from request and response logging.
