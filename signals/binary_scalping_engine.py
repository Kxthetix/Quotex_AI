"""Experimental candle-based trend-pullback signals for paper evaluation.

The rules combine trend alignment, a pullback, candle confirmation, momentum, and
an extension filter. The returned score measures rule confluence; it is not a
calibrated probability of winning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from data.ingestion.models import Candle
from signals.models import SignalAction


class BinaryAction(str, Enum):
    CALL = "CALL"          # Buy / Up
    PUT = "PUT"            # Sell / Down
    NO_TRADE = "NO_TRADE"  # Hold / Capital Preservation


@dataclass
class CandleGeometry:
    """Micro-structural geometry of a single candle."""
    open: float
    high: float
    low: float
    close: float
    body: float
    body_pct: float
    upper_wick: float
    lower_wick: float
    is_bullish: bool
    is_bearish: bool
    is_doji: bool
    is_hammer: bool
    is_shooting_star: bool


@dataclass
class TechnicalState:
    """Computed technical indicators for the current bar."""
    price: float
    ema_9: float
    ema_21: float
    ema_50: float | None
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    atr_14: float
    trend_state: str  # BULLISH, BEARISH, CONSOLIDATION
    momentum_state: str  # ACCELERATING, DECELERATING, OVERBOUGHT, OVERSOLD, NEUTRAL
    candle: CandleGeometry


@dataclass
class BinarySignalResult:
    """Decision and audit trail; confidence is rule confluence, not win probability."""
    action: BinaryAction
    confidence: float  # Retained for compatibility; not a calibrated probability.
    score: float  # [-1.0, +1.0]
    asset: str
    timestamp: datetime
    reasons: list[str]
    rejection_flags: list[str]
    technical_state: TechnicalState | None = None
    crowd_sentiment_applied: dict[str, Any] = field(default_factory=dict)


class BinaryScalpingEngine:
    """Rule-based candle strategy for research and paper evaluation.

    Combines:
    1. Triple EMA Trend Alignment (9, 21, 50)
    2. RSI-14 Mean-Reversion Exhaustion Filters (prevents buying tops / selling bottoms)
    3. MACD Momentum Histogram Confirmation
    4. Bollinger Bands Volatility Envelopes
    5. Candlestick Micro-Price-Action (Hammers, Shooting Stars, Dojis, Engulfing)
    6. Trend-pullback and candle-confirmation entry filter
    7. Optional crowd sentiment veto; sentiment never creates a trade
    """

    def __init__(
        self,
        min_confidence: float = 0.65,
        rsi_overbought: float = 68.0,
        rsi_oversold: float = 32.0,
        ema_separation_threshold: float = 0.0001,
    ) -> None:
        self.min_confidence = min_confidence
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.ema_separation_threshold = ema_separation_threshold

    @staticmethod
    def analyze_candle_geometry(c_open: float, c_high: float, c_low: float, c_close: float) -> CandleGeometry:
        """Analyze individual candlestick shape, wicks, and rejection patterns."""
        total_range = max(c_high - c_low, 1e-6)
        body = abs(c_close - c_open)
        body_pct = body / total_range

        if c_close >= c_open:
            upper_wick = c_high - c_close
            lower_wick = c_open - c_low
            is_bullish = True
            is_bearish = False
        else:
            upper_wick = c_high - c_open
            lower_wick = c_close - c_low
            is_bullish = False
            is_bearish = True

        is_doji = body_pct < 0.06
        # Hammer: long lower wick (>= 2x body) and small upper wick (<= body)
        is_hammer = (lower_wick >= 2.0 * body) and (upper_wick <= body) and not is_doji
        # Shooting star: long upper wick (>= 2x body) and small lower wick (<= body)
        is_shooting_star = (upper_wick >= 2.0 * body) and (lower_wick <= body) and not is_doji

        return CandleGeometry(
            open=c_open,
            high=c_high,
            low=c_low,
            close=c_close,
            body=body,
            body_pct=body_pct,
            upper_wick=upper_wick,
            lower_wick=lower_wick,
            is_bullish=is_bullish,
            is_bearish=is_bearish,
            is_doji=is_doji,
            is_hammer=is_hammer,
            is_shooting_star=is_shooting_star,
        )

    def calculate_technicals(self, df: pd.DataFrame) -> TechnicalState | None:
        """Compute full indicator set from an OHLCV DataFrame."""
        if len(df) < 25:
            return None

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        current_price = float(closes[-1])

        # 1. EMAs
        ema_9 = float(pd.Series(closes).ewm(span=9, adjust=False).mean().iloc[-1])
        ema_21 = float(pd.Series(closes).ewm(span=21, adjust=False).mean().iloc[-1])
        ema_50 = float(pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1]) if len(closes) >= 50 else None

        # Trend Determination
        separation = (ema_9 - ema_21) / current_price
        if abs(separation) < self.ema_separation_threshold:
            trend_state = "CONSOLIDATION"
        elif ema_9 > ema_21 and current_price >= ema_9 * 0.9995:
            trend_state = "BULLISH"
        elif ema_9 < ema_21 and current_price <= ema_9 * 1.0005:
            trend_state = "BEARISH"
        else:
            trend_state = "CONSOLIDATION"

        # 2. RSI (14)
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = pd.Series(gains).ewm(com=13, adjust=False).mean().iloc[-1]
        avg_loss = pd.Series(losses).ewm(com=13, adjust=False).mean().iloc[-1]
        if avg_loss == 0:
            rsi_14 = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_14 = float(100.0 - (100.0 / (1.0 + rs)))

        # 3. MACD (12, 26, 9)
        ema_12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
        ema_26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
        macd_series = ema_12 - ema_26
        signal_series = macd_series.ewm(span=9, adjust=False).mean()
        macd_hist_series = macd_series - signal_series

        macd_line = float(macd_series.iloc[-1])
        macd_sig = float(signal_series.iloc[-1])
        macd_hist = float(macd_hist_series.iloc[-1])

        # 4. Bollinger Bands (20, 2.0 std)
        sma_20 = float(pd.Series(closes).rolling(20).mean().iloc[-1])
        std_20 = float(pd.Series(closes).rolling(20).std().iloc[-1])
        bb_upper = sma_20 + (2.0 * std_20)
        bb_lower = sma_20 - (2.0 * std_20)

        previous_close = pd.Series(closes).shift(1)
        true_range = pd.concat(
            [
                pd.Series(highs - lows),
                pd.Series(np.abs(highs - previous_close.to_numpy())),
                pd.Series(np.abs(lows - previous_close.to_numpy())),
            ],
            axis=1,
        ).max(axis=1)
        atr_14 = float(true_range.ewm(alpha=1.0 / 14.0, adjust=False).mean().iloc[-1])

        # 5. Momentum classification
        if rsi_14 >= self.rsi_overbought:
            momentum_state = "OVERBOUGHT"
        elif rsi_14 <= self.rsi_oversold:
            momentum_state = "OVERSOLD"
        elif macd_hist > 0 and macd_hist > float(macd_hist_series.iloc[-2]):
            momentum_state = "ACCELERATING"
        elif macd_hist < 0 and macd_hist < float(macd_hist_series.iloc[-2]):
            momentum_state = "DECELERATING"
        else:
            momentum_state = "NEUTRAL"

        # 6. Current candle geometry
        last_open = float(df["open"].iloc[-1])
        last_high = float(highs[-1])
        last_low = float(lows[-1])
        last_close = current_price
        candle = self.analyze_candle_geometry(last_open, last_high, last_low, last_close)

        return TechnicalState(
            price=current_price,
            ema_9=round(ema_9, 5),
            ema_21=round(ema_21, 5),
            ema_50=round(ema_50, 5) if ema_50 is not None else None,
            rsi_14=round(rsi_14, 2),
            macd_line=round(macd_line, 6),
            macd_signal=round(macd_sig, 6),
            macd_hist=round(macd_hist, 6),
            bb_upper=round(bb_upper, 5),
            bb_middle=round(sma_20, 5),
            bb_lower=round(bb_lower, 5),
            atr_14=atr_14,
            trend_state=trend_state,
            momentum_state=momentum_state,
            candle=candle,
        )

    def evaluate(
        self,
        candles: list[Candle] | pd.DataFrame,
        asset: str = "EUR/CHF",
        crowd_sentiment_bullish_pct: float | None = None,
    ) -> BinarySignalResult:
        """Evaluate a completed-candle trend-pullback setup."""
        timestamp = datetime.now(UTC)

        if isinstance(candles, list):
            if candles:
                timestamp = candles[-1].end_time
            if any(not c.is_closed for c in candles):
                return BinarySignalResult(
                    action=BinaryAction.NO_TRADE,
                    confidence=0.0,
                    score=0.0,
                    asset=asset,
                    timestamp=timestamp,
                    reasons=["One or more candles in the series are incomplete/open; only completed candles permit evaluation."],
                    rejection_flags=["INCOMPLETE_CANDLE"],
                )
            df = pd.DataFrame([
                {"open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
                for c in candles
            ])
        else:
            df = candles.copy()
            if "is_closed" in df.columns and not df["is_closed"].all():
                return BinarySignalResult(
                    action=BinaryAction.NO_TRADE,
                    confidence=0.0,
                    score=0.0,
                    asset=asset,
                    timestamp=timestamp,
                    reasons=["One or more candles in the DataFrame are incomplete/open; only completed candles permit evaluation."],
                    rejection_flags=["INCOMPLETE_CANDLE"],
                )

        tech = self.calculate_technicals(df)
        if tech is None:
            return BinarySignalResult(
                action=BinaryAction.NO_TRADE,
                confidence=0.0,
                score=0.0,
                asset=asset,
                timestamp=timestamp,
                reasons=["Insufficient historical bars for indicator stabilization (<25 bars)."],
                rejection_flags=["INSUFFICIENT_DATA"],
            )

        score = 0.0
        reasons: list[str] = []
        rejection_flags: list[str] = []

        if tech.atr_14 <= 0.0 or not np.isfinite(tech.atr_14):
            rejection_flags.append("INVALID_VOLATILITY")

        # The setup is trend continuation after a measured pullback, not a single
        # indicator crossover. The final candle must confirm direction after the pullback.
        previous_candle = df.iloc[-2]
        pullback_window = df.iloc[-5:-1]
        atr = tech.atr_14
        pullback_long = bool(
            ((pullback_window["low"] <= tech.ema_21 + 0.25 * atr)
             & (pullback_window["low"] >= tech.ema_21 - 0.75 * atr)).any()
        )
        pullback_short = bool(
            ((pullback_window["high"] >= tech.ema_21 - 0.25 * atr)
             & (pullback_window["high"] <= tech.ema_21 + 0.75 * atr)).any()
        )
        candle_range = max(tech.candle.high - tech.candle.low, 1e-12)
        close_location = (tech.candle.close - tech.candle.low) / candle_range
        bullish_confirmation = (
            tech.candle.is_bullish
            and tech.candle.close > float(previous_candle["high"])
            and close_location >= 0.65
        )
        bearish_confirmation = (
            tech.candle.is_bearish
            and tech.candle.close < float(previous_candle["low"])
            and close_location <= 0.35
        )
        trend_up = tech.ema_9 > tech.ema_21 and (
            tech.ema_50 is None or tech.ema_21 > tech.ema_50
        )
        trend_down = tech.ema_9 < tech.ema_21 and (
            tech.ema_50 is None or tech.ema_21 < tech.ema_50
        )
        not_extended = abs(tech.price - tech.ema_21) <= 1.5 * atr if atr > 0 else False
        close_series = pd.Series(df["close"].values)
        macd_series = close_series.ewm(span=12, adjust=False).mean() - close_series.ewm(
            span=26, adjust=False
        ).mean()
        previous_macd_hist = float(
            (macd_series - macd_series.ewm(span=9, adjust=False).mean()).iloc[-2]
        )
        momentum_up = (
            50.0 <= tech.rsi_14 < self.rsi_overbought
            and tech.macd_hist > 0.0
            and tech.macd_hist >= previous_macd_hist
        )
        momentum_down = (
            self.rsi_oversold < tech.rsi_14 <= 50.0
            and tech.macd_hist < 0.0
            and tech.macd_hist <= previous_macd_hist
        )

        if not pullback_long and not pullback_short:
            rejection_flags.append("NO_PULLBACK_SETUP")
            reasons.append("No recent retest of the 21-period EMA; skip chasing price.")
        if not bullish_confirmation and not bearish_confirmation:
            rejection_flags.append("NO_CANDLE_CONFIRMATION")
            reasons.append("Latest candle did not close beyond the prior candle in either direction.")
        if not not_extended:
            rejection_flags.append("PRICE_EXTENDED_FROM_MEAN")
            reasons.append("Price is more than 1.5 ATR from the 21-period EMA.")
        if not trend_up and not trend_down:
            rejection_flags.append("TREND_CONSOLIDATION")
            reasons.append("Trend is sideways, consolidating, or EMA hierarchy is misaligned.")
        if not momentum_up and not momentum_down:
            rejection_flags.append("MOMENTUM_MISALIGNED")
            reasons.append("Momentum indicators (RSI boundary or MACD histogram slope) are misaligned with trend.")

        # -------------------------------------------------------------
        # LAYER 1: Trend Alignment (Weight: 0.35)
        # -------------------------------------------------------------
        if tech.trend_state == "BULLISH":
            score += 0.35
            reasons.append(f"Trend Bullish: EMA-9 ({tech.ema_9}) > EMA-21 ({tech.ema_21}) and price holds above.")
        elif tech.trend_state == "BEARISH":
            score -= 0.35
            reasons.append(f"Trend Bearish: EMA-9 ({tech.ema_9}) < EMA-21 ({tech.ema_21}) and price holds below.")
        else:
            reasons.append("Trend Sideways/Consolidation: EMA separation below significance threshold.")

        # -------------------------------------------------------------
        # LAYER 2: RSI Exhaustion & Mean Reversion Filter (Weight: 0.25)
        # -------------------------------------------------------------
        if tech.rsi_14 >= self.rsi_overbought:
            rejection_flags.append("RSI_OVERBOUGHT")
            reasons.append(f"RSI Exhaustion: RSI={tech.rsi_14:.1f} >= {self.rsi_overbought:.1f}. High pullback danger.")
            # Penalize buying; overbought encourages contrarian put or no-trade
            score -= 0.30
        elif tech.rsi_14 <= self.rsi_oversold:
            rejection_flags.append("RSI_OVERSOLD")
            reasons.append(f"RSI Exhaustion: RSI={tech.rsi_14:.1f} <= {self.rsi_oversold:.1f}. High bounce potential.")
            # Penalize selling; oversold encourages contrarian call or no-trade
            score += 0.30
        else:
            # Healthy trend continuation zone
            if 45.0 <= tech.rsi_14 <= 62.0 and tech.trend_state == "BULLISH":
                score += 0.20
                reasons.append(f"RSI Healthy Momentum: RSI={tech.rsi_14:.1f} supports upward expansion.")
            elif 38.0 <= tech.rsi_14 <= 55.0 and tech.trend_state == "BEARISH":
                score -= 0.20
                reasons.append(f"RSI Healthy Momentum: RSI={tech.rsi_14:.1f} supports downward expansion.")

        # -------------------------------------------------------------
        # LAYER 3: Bollinger Bands Position (Weight: 0.15)
        # -------------------------------------------------------------
        if tech.price >= tech.bb_upper:
            rejection_flags.append("BB_UPPER_PIERCED")
            reasons.append(f"Bollinger Band Resistance: Price ({tech.price}) pierced upper band ({tech.bb_upper}).")
            score -= 0.15
        elif tech.price <= tech.bb_lower:
            rejection_flags.append("BB_LOWER_PIERCED")
            reasons.append(f"Bollinger Band Support: Price ({tech.price}) pierced lower band ({tech.bb_lower}).")
            score += 0.15

        # -------------------------------------------------------------
        # LAYER 4: MACD Momentum (Weight: 0.15)
        # -------------------------------------------------------------
        if tech.macd_hist > 0:
            score += 0.15
            reasons.append(f"MACD Bullish Momentum: Histogram positive ({tech.macd_hist:.6f}).")
        elif tech.macd_hist < 0:
            score -= 0.15
            reasons.append(f"MACD Bearish Momentum: Histogram negative ({tech.macd_hist:.6f}).")

        # -------------------------------------------------------------
        # LAYER 5: Candlestick Micro-Price-Action (Weight: 0.15)
        # -------------------------------------------------------------
        if tech.candle.is_hammer:
            score += 0.20
            reasons.append("Bullish Rejection: Hammer/Pin-bar detected with strong lower wick buying defense.")
        elif tech.candle.is_shooting_star:
            score -= 0.20
            reasons.append("Bearish Rejection: Shooting Star detected with strong upper wick selling defense.")
        elif tech.candle.is_doji:
            rejection_flags.append("DOJI_INDECISION")
            reasons.append("Indecision: Doji candle detected. Market in micro-equilibrium.")
        elif tech.candle.is_bullish and tech.candle.body_pct > 0.60:
            score += 0.10
            reasons.append(f"Bullish Body Expansion: Solid green candle ({tech.candle.body_pct*100:.0f}% body).")
        elif tech.candle.is_bearish and tech.candle.body_pct > 0.60:
            score -= 0.10
            reasons.append(f"Bearish Body Expansion: Solid red candle ({tech.candle.body_pct*100:.0f}% body).")

        # -------------------------------------------------------------
        # LAYER 6: Contrarian Crowd Sentiment Shield
        # -------------------------------------------------------------
        sentiment_info: dict[str, Any] = {}
        if crowd_sentiment_bullish_pct is not None:
            sentiment_info["bullish_pct"] = crowd_sentiment_bullish_pct
            sentiment_info["bearish_pct"] = 100.0 - crowd_sentiment_bullish_pct

            # If crowd is overwhelmingly bullish (>85%) but price is overbought or stalling:
            if crowd_sentiment_bullish_pct >= 85.0:
                if "RSI_OVERBOUGHT" in rejection_flags or "BB_UPPER_PIERCED" in rejection_flags or tech.trend_state != "BULLISH":
                    rejection_flags.append("CROWD_TRAP_BULLISH")
                    reasons.append(
                        f"CONTRARIAN WARNING: Retail crowd is {crowd_sentiment_bullish_pct:.0f}% Bullish at resistance. Classic bull-trap!"
                    )
                    score -= 0.25  # Invert or penalize heavily
            elif crowd_sentiment_bullish_pct <= 15.0:
                if "RSI_OVERSOLD" in rejection_flags or "BB_LOWER_PIERCED" in rejection_flags or tech.trend_state != "BEARISH":
                    rejection_flags.append("CROWD_TRAP_BEARISH")
                    reasons.append(
                        f"CONTRARIAN WARNING: Retail crowd is {100.0 - crowd_sentiment_bullish_pct:.0f}% Bearish at support. Classic bear-trap!"
                    )
                    score += 0.25  # Invert or penalize heavily

        # Clamp composite score to [-1.0, 1.0]
        score = max(-1.0, min(1.0, score))
        confidence = abs(score)

        # -------------------------------------------------------------
        # DECISION LOGIC
        # -------------------------------------------------------------
        # Critical hard blocks:
        # Never CALL if RSI is overbought or if crowd trap detected
        can_call = (
            score >= self.min_confidence
            and trend_up
            and pullback_long
            and bullish_confirmation
            and momentum_up
            and not_extended
            and "INVALID_VOLATILITY" not in rejection_flags
            and "RSI_OVERBOUGHT" not in rejection_flags
            and "CROWD_TRAP_BULLISH" not in rejection_flags
            and "BB_UPPER_PIERCED" not in rejection_flags
            and "DOJI_INDECISION" not in rejection_flags
            and tech.trend_state == "BULLISH"
        )

        can_put = (
            score <= -self.min_confidence
            and trend_down
            and pullback_short
            and bearish_confirmation
            and momentum_down
            and not_extended
            and "INVALID_VOLATILITY" not in rejection_flags
            and "RSI_OVERSOLD" not in rejection_flags
            and "CROWD_TRAP_BEARISH" not in rejection_flags
            and "BB_LOWER_PIERCED" not in rejection_flags
            and "DOJI_INDECISION" not in rejection_flags
            and tech.trend_state == "BEARISH"
        )

        if can_call:
            action = BinaryAction.CALL
        elif can_put:
            action = BinaryAction.PUT
        else:
            action = BinaryAction.NO_TRADE
            if confidence < self.min_confidence and "LOW_CONFIDENCE" not in rejection_flags:
                rejection_flags.append("LOW_CONFIDENCE")
                reasons.append(f"Confidence {confidence:.2f} below minimum threshold ({self.min_confidence:.2f}).")
            if not rejection_flags:
                rejection_flags.append("NO_TRADE_SETUP")
                reasons.append("Rule confluence did not satisfy all trade entry conditions.")

        return BinarySignalResult(
            action=action,
            confidence=round(confidence, 4),
            score=round(score, 4),
            asset=asset,
            timestamp=timestamp,
            reasons=reasons,
            rejection_flags=rejection_flags,
            technical_state=tech,
            crowd_sentiment_applied=sentiment_info,
        )
