# Market Regime Engine Specification

## 1. Overview & Principles

The `MarketRegimeEngine` (version `v1.0.0`) classifies the prevailing market condition into distinct, objective market regimes. It begins with robust statistical and technical rules, deliberately keeping learned statistical models separate to establish a transparent, unadulterated baseline.

## 2. Regime Classifications

The engine supports 8 canonical market regimes:
1. `BREAKOUT`: Immediate expansion beyond historical local highs or lows ($breakout_{20} == 1$ or $breakdown_{20} == 1$).
2. `REVERSAL`: Exhaustion wick at oscillator extremes (e.g., upper wick $\ge 50\%$ of candle range with $RSI_{14} \ge 72$, or lower wick $\ge 50\%$ with $RSI_{14} \le 28$).
3. `CONSOLIDATION`: Compressing trading range with narrow range-compression metric ($range\_compression_{20} < 0.02$).
4. `HIGH_VOLATILITY`: Annualized realized volatility exceeding $1.6\times$ historical volatility.
5. `TREND_UP`: Directional upward trend ($EMA_{20} > EMA_{50}$ and $ADX_{14} \ge 25.0$ and $+DI_{14} > -DI_{14}$).
6. `TREND_DOWN`: Directional downward trend ($EMA_{20} < EMA_{50}$ and $ADX_{14} \ge 25.0$ and $-DI_{14} > +DI_{14}$).
7. `LOW_VOLATILITY`: Annualized realized volatility falling below $0.65\times$ historical volatility.
8. `SIDEWAYS`: Range-bound market with low trend strength ($ADX_{14} < 20.0$).

## 3. Deterministic Precedence Resolution

When market indicators satisfy multiple regime conditions simultaneously, the engine applies a strict deterministic precedence order:

$$\text{BREAKOUT} > \text{REVERSAL} > \text{CONSOLIDATION} > \text{HIGH\_VOLATILITY} > \text{TREND\_UP} / \text{TREND\_DOWN} > \text{LOW\_VOLATILITY} > \text{SIDEWAYS}$$

In addition to the primary resolved regime, the `RegimeAssessment` payload captures `active_subregimes` and numerical evidence scores to preserve structural context for subsequent ML and pattern engines.
