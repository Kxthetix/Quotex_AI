# Technical Rule Engine Specification

## 1. Overview & Architecture

The `TechnicalRuleEngine` (version `v1.0.0`) operates on Phase 3 `FeatureVector` payloads at time $T$. It applies deterministic, rule-based heuristics across core market dimensions to generate an auditable, structured technical assessment without subjective or discretionary trading opinions.

```
+--------------------+
|   FeatureVector    | (Produced from candles <= T)
+--------------------+
          |
          v
+--------------------+
| TechnicalRuleEngine| (Evaluates trend, momentum, volatility, volume, structure)
+--------------------+
          |
          v
+--------------------+
|TechnicalAssessment | (Deterministic observation ledger with audit evidence)
+--------------------+
```

## 2. Technical States & Enumerations

### 2.1 Trend State (`TrendState`)
- `BULLISH`: EMA alignment $EMA_{20} > EMA_{50}$ with directional strength ($ADX_{14} \ge 25.0$ and $+DI_{14} > -DI_{14}$).
- `BEARISH`: EMA alignment $EMA_{20} < EMA_{50}$ with directional strength ($ADX_{14} \ge 25.0$ and $-DI_{14} > +DI_{14}$).
- `NEUTRAL`: Non-trending or weak directional conviction ($ADX_{14} < 25.0$).

### 2.2 Momentum State (`MomentumState`)
- `OVERBOUGHT`: $RSI_{14} \ge 70.0$.
- `OVERSOLD`: $RSI_{14} \le 30.0$.
- `BULLISH`: $RSI_{14} > 50.0$ and $MACD_{hist} > 0$.
- `BEARISH`: $RSI_{14} < 50.0$ and $MACD_{hist} < 0$.
- `NEUTRAL`: Oscillators diverging or near midpoint.

### 2.3 Volatility State (`VolatilityState`)
- `COMPRESSING`: $range\_compression_{20} < 0.015$ (tight consolidation / squeeze).
- `EXPANDING`: $rolling\_vol_{20} \times 500 > historical\_vol_{20}$ (volatility surge).
- `NORMAL`: Standard baseline range.

### 2.4 Volume State (`VolumeState`)
- `SPIKE`: Volume exceeds 3 standard deviations ($volume\_spike == 1.0$).
- `HIGH`: Relative 20-bar volume $> 1.5$.
- `LOW`: Relative 20-bar volume $< 0.6$.
- `NORMAL`: Baseline volume between $0.6$ and $1.5$.

### 2.5 Structure State (`StructureState`)
- `BREAKOUT_UP`: 20-bar high breakout ($breakout_{20} == 1.0$).
- `BREAKDOWN`: 20-bar low breakdown ($breakdown_{20} == 1.0$).
- `CONSOLIDATION`: Active 20-bar consolidation channel ($is\_consolidating_{20} == 1.0$).
- `NORMAL`: Normal structured movement.

### 2.6 Multi-Timeframe Alignment (`MultiTimeframeState`)
- `ALIGNED_BULLISH`: Confluence across 1m, 5m, and 15m trend features.
- `ALIGNED_BEARISH`: Confluence across 1m, 5m, and 15m trend features.
- `CONFLICTING`: Divergent trend directions across timeframes.
- `INSUFFICIENT_DATA`: Multi-timeframe warmup incomplete.

## 3. Versioning Invariant
The rule version is locked to `v1.0.0`. Modifying any threshold, formula, or interpretation requires a semver bump (e.g., `v1.1.0`), preventing silent mutation of historical evaluations.
