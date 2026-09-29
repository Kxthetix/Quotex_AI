# Microstructure Features & Leakage Protection

## 1. Feature Set: `orderbook_features_v1`

| Feature | Formula | Microstructure Significance |
| :--- | :--- | :--- |
| **`bid_ask_spread`** | $P_{\text{ask}, 1} - P_{\text{bid}, 1}$ | Direct measure of liquidity and immediacy cost. |
| **`relative_spread`** | $\frac{P_{\text{ask}, 1} - P_{\text{bid}, 1}}{P_{\text{mid}}}$ | Dimensionless spread normalized by price level. |
| **`top_of_book_imbalance`**| $\frac{Q_{\text{bid}, 1} - Q_{\text{ask}, 1}}{Q_{\text{bid}, 1} + Q_{\text{ask}, 1}}$ | Instantaneous buying vs selling queue pressure. |
| **`n_level_imbalance`** | $\frac{\sum_{i=1}^N Q_{\text{bid}, i} - \sum_{i=1}^N Q_{\text{ask}, i}}{\sum_{i=1}^N Q_{\text{bid}, i} + \sum_{i=1}^N Q_{\text{ask}, i}}$ | Medium-depth volume imbalance across top $N$ levels ($N=5$). |
| **`depth_weighted_imbalance`** | $\frac{\sum \frac{Q_{\text{bid}, i}}{i} - \sum \frac{Q_{\text{ask}, i}}{i}}{\sum \frac{Q_{\text{bid}, i}}{i} + \sum \frac{Q_{\text{ask}, i}}{i}}$ | Distance-decay weighted queue imbalance. |
| **`microprice`** | $\frac{P_{\text{ask}, 1} Q_{\text{bid}, 1} + P_{\text{bid}, 1} Q_{\text{ask}, 1}}{Q_{\text{bid}, 1} + Q_{\text{ask}, 1}}$ | Volume-weighted equilibrium price. |
| **`microprice_deviation`**| $\text{microprice} - P_{\text{mid}}$ | Expected short-term price drift. |
| **`book_pressure`** | $\frac{\sum Q_{\text{bid}}}{\sum Q_{\text{bid}} + \sum Q_{\text{ask}}}$ | Overall book depth supply vs demand ratio. |
| **`liquidity_concentration`** | $\frac{\sum_{i=1}^3 (Q_{\text{bid}, i} + Q_{\text{ask}, i})}{\text{Total Depth Volume}}$ | Clustered liquidity near top of book. |
| **`depth_slope`** | $\frac{(Q_N - Q_1)}{(N - 1)}$ | Rate of volume replenishment away from touch. |

## 2. Temporal Leakage Protection (Section 14)

Section 14 mandates that features at time $T$ strictly utilize information available at or before $T$:
- If `as_of_time` is specified and `snapshot.timestamp > as_of_time`, the calculation immediately raises `ValueError` to block forward-looking data leakage.
