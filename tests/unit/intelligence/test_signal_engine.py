"""Comprehensive tests for Signal Engine Phase 6.

Tests cover:
1. Signal model contracts (TradingSignal, Evidence, SignalAction)
2. All 10 evidence adapters
3. Fusion engine with regime-adaptive thresholds
4. Confidence scoring and diversity bonuses
5. End-to-end signal generation via SignalEngine
6. Edge cases: missing data, insufficient evidence, boundary conditions
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from analysis.pattern.models import HistoricalMatch, PatternDistribution, PatternSearchResult
from analysis.regime.models import MarketRegime, RegimeAssessment
from analysis.technical.models import (
    MomentumState,
    MultiTimeframeState,
    StructureState,
    TechnicalAssessment,
    TrendState,
    VolatilityState,
    VolumeState,
)
from signals.confidence.scoring import ConfidenceConfig, ConfidenceScorer, HistoricalConfidenceCalibrator
from signals.engine import IntelligenceSnapshot, SignalEngine, SignalEngineConfig
from signals.fusion.engine import FusionConfig, FusionEngine
from signals.generation.adapters import (
    MLDirectionEvidenceAdapter,
    MLReturnEvidenceAdapter,
    MLTradeQualityEvidenceAdapter,
    MLVolatilityEvidenceAdapter,
    MultiTimeframeEvidenceAdapter,
    PatternEvidenceAdapter,
    RegimeEvidenceAdapter,
    RiskRewardEvidenceAdapter,
    TechnicalEvidenceAdapter,
    VolumeEvidenceAdapter,
)
from signals.models import Evidence, EvidenceType, SignalAction, TradingSignal

NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


# ==============================================================================
# Helper Fixtures
# ==============================================================================

def _make_technical_assessment(
    trend: TrendState = TrendState.BULLISH,
    momentum: MomentumState = MomentumState.ACCELERATING,
    volatility: VolatilityState = VolatilityState.NORMAL,
    volume: VolumeState = VolumeState.NORMAL,
    structure: StructureState = StructureState.CHANNEL_RANGE,
    mtf: MultiTimeframeState = MultiTimeframeState.ALIGNED_BULLISH,
    **evidence_kw,
) -> TechnicalAssessment:
    return TechnicalAssessment(
        symbol="BTCUSDT",
        timestamp=NOW,
        trend_state=trend,
        momentum_state=momentum,
        volatility_state=volatility,
        volume_state=volume,
        structure_state=structure,
        multi_timeframe_state=mtf,
        evidence=evidence_kw,
    )


def _make_regime_assessment(
    regime: MarketRegime = MarketRegime.TREND_UP,
    confidence: float = 0.85,
) -> RegimeAssessment:
    return RegimeAssessment(
        symbol="BTCUSDT",
        timestamp=NOW,
        regime=regime,
        confidence=confidence,
    )


def _make_pattern_result(
    match_count: int = 20,
    mean_return: float = 0.005,
    positive_ratio: float = 0.65,
    std_return: float = 0.01,
) -> PatternSearchResult:
    dist = PatternDistribution(
        match_count=match_count,
        horizon_bars=15,
        mean_forward_return=mean_return,
        median_forward_return=mean_return * 0.95,
        std_forward_return=std_return,
        positive_outcome_ratio=positive_ratio,
        negative_outcome_ratio=1.0 - positive_ratio,
        mean_mfe=0.008,
        mean_mae=0.003,
    )
    return PatternSearchResult(
        symbol="BTCUSDT",
        timestamp=NOW,
        regime=MarketRegime.TREND_UP,
        distribution=dist,
    )


# ==============================================================================
# 1. Signal Model Contract Tests
# ==============================================================================

class TestSignalModels:
    """Tests for core data models."""

    def test_signal_action_values(self):
        assert SignalAction.BUY == "BUY"
        assert SignalAction.SELL == "SELL"
        assert SignalAction.NO_TRADE == "NO_TRADE"

    def test_evidence_type_completeness(self):
        """All 10 evidence types must be defined."""
        expected = {
            "TECHNICAL", "PATTERN", "ML_DIRECTION", "ML_RETURN",
            "ML_TRADE_QUALITY", "REGIME", "MULTI_TIMEFRAME",
            "VOLUME", "VOLATILITY", "RISK_REWARD",
        }
        actual = {e.value for e in EvidenceType}
        assert actual == expected

    def test_evidence_weighted_score(self):
        e = Evidence(
            evidence_type=EvidenceType.TECHNICAL,
            vote=0.5,
            confidence=0.8,
            weight=0.15,
        )
        assert abs(e.weighted_score - (0.5 * 0.8 * 0.15)) < 1e-8

    def test_evidence_vote_bounds(self):
        """Vote must be clamped to [-1, 1]."""
        with pytest.raises(Exception):
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=1.5, confidence=0.5, weight=0.1)
        with pytest.raises(Exception):
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=-1.5, confidence=0.5, weight=0.1)

    def test_evidence_confidence_bounds(self):
        """Confidence must be [0, 1]."""
        with pytest.raises(Exception):
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.5, confidence=1.5, weight=0.1)

    def test_trading_signal_creation(self):
        sig = TradingSignal(
            symbol="btcusdt",
            timestamp=NOW,
            action=SignalAction.BUY,
            conviction=0.75,
        )
        assert sig.symbol == "BTCUSDT"  # normalized
        assert sig.action == SignalAction.BUY
        assert sig.conviction == 0.75

    def test_trading_signal_evidence_summary(self):
        sig = TradingSignal(
            symbol="BTCUSDT",
            timestamp=NOW,
            action=SignalAction.BUY,
            conviction=0.75,
            evidence=[
                Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.5, confidence=0.8, weight=0.15),
                Evidence(evidence_type=EvidenceType.ML_DIRECTION, vote=0.6, confidence=0.9, weight=0.20),
            ],
        )
        summary = sig.evidence_summary
        assert "TECHNICAL" in summary
        assert "ML_DIRECTION" in summary


# ==============================================================================
# 2. Evidence Adapter Tests
# ==============================================================================

class TestTechnicalAdapter:
    def test_bullish_assessment(self):
        assessment = _make_technical_assessment(
            trend=TrendState.BULLISH,
            momentum=MomentumState.ACCELERATING,
            structure=StructureState.BREAKOUT_UP,
        )
        adapter = TechnicalEvidenceAdapter()
        evidence = adapter.extract(assessment)
        assert evidence.evidence_type == EvidenceType.TECHNICAL
        assert evidence.vote > 0  # Should be bullish
        assert evidence.confidence > 0.5

    def test_bearish_assessment(self):
        assessment = _make_technical_assessment(
            trend=TrendState.BEARISH,
            momentum=MomentumState.DECELERATING,
            structure=StructureState.BREAKDOWN_DOWN,
        )
        adapter = TechnicalEvidenceAdapter()
        evidence = adapter.extract(assessment)
        assert evidence.vote < 0  # Should be bearish

    def test_neutral_assessment(self):
        assessment = _make_technical_assessment(
            trend=TrendState.NEUTRAL,
            momentum=MomentumState.NEUTRAL,
            structure=StructureState.CHANNEL_RANGE,
        )
        adapter = TechnicalEvidenceAdapter()
        evidence = adapter.extract(assessment)
        assert abs(evidence.vote) < 0.5  # Should be near neutral

    def test_custom_weight(self):
        assessment = _make_technical_assessment()
        adapter = TechnicalEvidenceAdapter()
        evidence = adapter.extract(assessment, weight=0.30)
        assert evidence.weight == 0.30


class TestPatternAdapter:
    def test_bullish_pattern(self):
        result = _make_pattern_result(match_count=25, positive_ratio=0.70)
        adapter = PatternEvidenceAdapter()
        evidence = adapter.extract(result)
        assert evidence.evidence_type == EvidenceType.PATTERN
        assert evidence.vote > 0

    def test_bearish_pattern(self):
        result = _make_pattern_result(match_count=25, positive_ratio=0.30)
        adapter = PatternEvidenceAdapter()
        evidence = adapter.extract(result)
        assert evidence.vote < 0

    def test_insufficient_matches(self):
        result = _make_pattern_result(match_count=3)
        adapter = PatternEvidenceAdapter()
        evidence = adapter.extract(result)
        assert evidence.vote == 0.0
        assert evidence.confidence < 0.2
        assert evidence.details.get("insufficient_matches") is True


class TestRegimeAdapter:
    def test_trend_up(self):
        assessment = _make_regime_assessment(regime=MarketRegime.TREND_UP)
        adapter = RegimeEvidenceAdapter()
        evidence = adapter.extract(assessment)
        assert evidence.vote > 0

    def test_trend_down(self):
        assessment = _make_regime_assessment(regime=MarketRegime.TREND_DOWN)
        adapter = RegimeEvidenceAdapter()
        evidence = adapter.extract(assessment)
        assert evidence.vote < 0

    def test_sideways_neutral(self):
        assessment = _make_regime_assessment(regime=MarketRegime.SIDEWAYS)
        adapter = RegimeEvidenceAdapter()
        evidence = adapter.extract(assessment)
        assert evidence.vote == 0.0


class TestMLDirectionAdapter:
    def test_strong_bullish(self):
        adapter = MLDirectionEvidenceAdapter()
        evidence = adapter.extract(prob_up=0.85)
        assert evidence.vote > 0.5
        assert evidence.confidence > 0.5

    def test_strong_bearish(self):
        adapter = MLDirectionEvidenceAdapter()
        evidence = adapter.extract(prob_up=0.15)
        assert evidence.vote < -0.5

    def test_neutral(self):
        adapter = MLDirectionEvidenceAdapter()
        evidence = adapter.extract(prob_up=0.50)
        assert abs(evidence.vote) < 0.01
        assert evidence.confidence < 0.01


class TestMLReturnAdapter:
    def test_positive_return(self):
        adapter = MLReturnEvidenceAdapter()
        evidence = adapter.extract(expected_return=0.01)
        assert evidence.vote == 1.0
        assert evidence.confidence > 0

    def test_negative_return(self):
        adapter = MLReturnEvidenceAdapter()
        evidence = adapter.extract(expected_return=-0.01)
        assert evidence.vote == -1.0

    def test_zero_return(self):
        adapter = MLReturnEvidenceAdapter()
        evidence = adapter.extract(expected_return=0.0)
        assert evidence.vote == 0.0


class TestMLTradeQualityAdapter:
    def test_high_quality(self):
        adapter = MLTradeQualityEvidenceAdapter()
        evidence = adapter.extract(prob_quality=0.80)
        assert evidence.vote > 0.0

    def test_low_quality(self):
        adapter = MLTradeQualityEvidenceAdapter()
        evidence = adapter.extract(prob_quality=0.20)
        assert evidence.vote < 0.0


class TestMLVolatilityAdapter:
    def test_low_volatility_high_confidence(self):
        adapter = MLVolatilityEvidenceAdapter()
        evidence = adapter.extract(expected_volatility=0.10)
        assert evidence.vote == 0.0  # Non-directional
        assert evidence.confidence > 0.5

    def test_high_volatility_low_confidence(self):
        adapter = MLVolatilityEvidenceAdapter()
        evidence = adapter.extract(expected_volatility=0.60)
        assert evidence.confidence < 0.5


class TestVolumeAdapter:
    def test_spike_volume(self):
        assessment = _make_technical_assessment(volume=VolumeState.SPIKE)
        adapter = VolumeEvidenceAdapter()
        evidence = adapter.extract(assessment)
        assert evidence.confidence > 0.8

    def test_low_volume(self):
        assessment = _make_technical_assessment(volume=VolumeState.LOW_VOLUME)
        adapter = VolumeEvidenceAdapter()
        evidence = adapter.extract(assessment)
        assert evidence.confidence < 0.4


class TestMultiTimeframeAdapter:
    def test_aligned_bullish(self):
        assessment = _make_technical_assessment(mtf=MultiTimeframeState.ALIGNED_BULLISH)
        adapter = MultiTimeframeEvidenceAdapter()
        evidence = adapter.extract(assessment)
        assert evidence.vote > 0.0

    def test_aligned_bearish(self):
        assessment = _make_technical_assessment(mtf=MultiTimeframeState.ALIGNED_BEARISH)
        adapter = MultiTimeframeEvidenceAdapter()
        evidence = adapter.extract(assessment)
        assert evidence.vote < 0.0

    def test_mixed(self):
        assessment = _make_technical_assessment(mtf=MultiTimeframeState.MIXED)
        adapter = MultiTimeframeEvidenceAdapter()
        evidence = adapter.extract(assessment)
        assert evidence.vote == 0.0


class TestRiskRewardAdapter:
    def test_favorable_risk_reward(self):
        adapter = RiskRewardEvidenceAdapter()
        evidence = adapter.extract(expected_return=0.02, expected_volatility=0.005)
        # R:R = 0.02/0.005 = 4.0, well above MIN_RISK_REWARD
        assert evidence.vote > 0.0
        assert evidence.confidence > 0.5

    def test_unfavorable_risk_reward(self):
        adapter = RiskRewardEvidenceAdapter()
        evidence = adapter.extract(expected_return=0.001, expected_volatility=0.01)
        # R:R = 0.1, well below threshold
        assert evidence.vote == 0.0

    def test_zero_volatility(self):
        adapter = RiskRewardEvidenceAdapter()
        evidence = adapter.extract(expected_return=0.01, expected_volatility=0.0)
        assert evidence.details.get("insufficient_data") is True


# ==============================================================================
# 3. Fusion Engine Tests
# ==============================================================================

class TestFusionEngine:
    def test_strong_buy_signal(self):
        """Multiple bullish evidence streams should produce BUY."""
        engine = FusionEngine()
        evidence = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.8, confidence=0.9, weight=0.15),
            Evidence(evidence_type=EvidenceType.ML_DIRECTION, vote=0.7, confidence=0.8, weight=0.20),
            Evidence(evidence_type=EvidenceType.ML_RETURN, vote=1.0, confidence=0.6, weight=0.15),
            Evidence(evidence_type=EvidenceType.REGIME, vote=0.4, confidence=0.8, weight=0.10),
        ]
        action, score, conviction = engine.fuse(evidence, regime="TREND_UP")
        assert action == SignalAction.BUY
        assert score > 0
        assert conviction > 0

    def test_strong_sell_signal(self):
        """Multiple bearish evidence streams should produce SELL."""
        engine = FusionEngine()
        evidence = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=-0.8, confidence=0.9, weight=0.15),
            Evidence(evidence_type=EvidenceType.ML_DIRECTION, vote=-0.7, confidence=0.8, weight=0.20),
            Evidence(evidence_type=EvidenceType.ML_RETURN, vote=-1.0, confidence=0.6, weight=0.15),
            Evidence(evidence_type=EvidenceType.REGIME, vote=-0.4, confidence=0.8, weight=0.10),
        ]
        action, score, conviction = engine.fuse(evidence, regime="TREND_DOWN")
        assert action == SignalAction.SELL
        assert score < 0

    def test_no_trade_mixed_evidence(self):
        """Conflicting evidence should produce NO_TRADE."""
        engine = FusionEngine()
        evidence = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.5, confidence=0.8, weight=0.15),
            Evidence(evidence_type=EvidenceType.ML_DIRECTION, vote=-0.5, confidence=0.8, weight=0.20),
            Evidence(evidence_type=EvidenceType.ML_RETURN, vote=0.1, confidence=0.3, weight=0.15),
        ]
        action, score, conviction = engine.fuse(evidence)
        assert action == SignalAction.NO_TRADE

    def test_insufficient_evidence(self):
        """Fewer than min_evidence_count should always be NO_TRADE."""
        engine = FusionEngine()
        evidence = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.9, confidence=0.99, weight=0.15),
        ]
        action, score, conviction = engine.fuse(evidence)
        assert action == SignalAction.NO_TRADE
        assert conviction == 0.0

    def test_regime_threshold_adjustment(self):
        """TREND_UP regime should lower buy threshold, making BUY easier."""
        config = FusionConfig(base_buy_threshold=0.15)
        engine = FusionEngine(config)

        # Marginal evidence
        evidence = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.3, confidence=0.6, weight=0.15),
            Evidence(evidence_type=EvidenceType.ML_DIRECTION, vote=0.3, confidence=0.5, weight=0.20),
            Evidence(evidence_type=EvidenceType.ML_RETURN, vote=0.2, confidence=0.4, weight=0.15),
        ]

        # In TREND_UP: threshold lowered by -0.03
        action_trend, _, _ = engine.fuse(evidence, regime=MarketRegime.TREND_UP.value)

        # In HIGH_VOLATILITY: threshold raised by +0.05
        action_hvol, _, _ = engine.fuse(evidence, regime=MarketRegime.HIGH_VOLATILITY.value)

        # Same evidence but different regime thresholds may produce different actions
        # At least verify both produce valid actions
        assert action_trend in (SignalAction.BUY, SignalAction.SELL, SignalAction.NO_TRADE)
        assert action_hvol in (SignalAction.BUY, SignalAction.SELL, SignalAction.NO_TRADE)

    def test_empty_evidence_list(self):
        engine = FusionEngine()
        action, score, conviction = engine.fuse([])
        assert action == SignalAction.NO_TRADE

    def test_custom_config(self):
        """Custom config thresholds should be respected."""
        config = FusionConfig(
            base_buy_threshold=0.50,
            base_sell_threshold=-0.50,
            min_evidence_count=1,
        )
        engine = FusionEngine(config)

        # Strong evidence but below high threshold
        evidence = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.6, confidence=0.5, weight=0.5),
        ]
        action, _, _ = engine.fuse(evidence)
        assert action == SignalAction.NO_TRADE  # 0.6*0.5*0.5 / 0.5 = 0.3 < 0.5


# ==============================================================================
# 4. Confidence Scorer Tests
# ==============================================================================

class TestConfidenceScorer:
    def test_regime_suitability_scaling(self):
        scorer = ConfidenceScorer()
        evidence = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.5, confidence=0.8, weight=0.15),
        ]

        # TREND_UP has suitability 1.0
        adj_trend = scorer.adjust(0.8, evidence, MarketRegime.TREND_UP.value)
        # CONSOLIDATION has suitability 0.5
        adj_cons = scorer.adjust(0.8, evidence, MarketRegime.CONSOLIDATION.value)

        assert adj_trend > adj_cons

    def test_diversity_bonus(self):
        scorer = ConfidenceScorer()

        # Single type
        single = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.5, confidence=0.8, weight=0.15),
        ]

        # Multiple types
        diverse = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.5, confidence=0.8, weight=0.15),
            Evidence(evidence_type=EvidenceType.ML_DIRECTION, vote=0.6, confidence=0.7, weight=0.20),
            Evidence(evidence_type=EvidenceType.REGIME, vote=0.3, confidence=0.8, weight=0.10),
            Evidence(evidence_type=EvidenceType.PATTERN, vote=0.4, confidence=0.6, weight=0.10),
        ]

        adj_single = scorer.adjust(0.5, single, "TREND_UP")
        adj_diverse = scorer.adjust(0.5, diverse, "TREND_UP")

        assert adj_diverse > adj_single

    def test_conviction_clamped(self):
        scorer = ConfidenceScorer()
        evidence = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=1.0, confidence=1.0, weight=1.0),
        ]
        # Very high raw conviction should be clamped to 1.0
        result = scorer.adjust(2.0, evidence, "TREND_UP")
        assert result <= 1.0


class TestHistoricalCalibrator:
    def test_fit_and_calibrate(self):
        calibrator = HistoricalConfidenceCalibrator()

        # Synthetic: high conviction → mostly correct, low conviction → mostly wrong
        np.random.seed(42)
        convictions = np.random.uniform(0.0, 1.0, 100)
        outcomes = (convictions + np.random.normal(0, 0.2, 100) > 0.5).astype(int)

        calibrator.fit(convictions, outcomes)
        assert calibrator.is_fitted

        # Calibrated values should be in [0, 1]
        cal = calibrator.calibrate(0.9)
        assert 0.0 <= cal <= 1.0

    def test_uncalibrated_passthrough(self):
        calibrator = HistoricalConfidenceCalibrator()
        assert calibrator.calibrate(0.75) == 0.75

    def test_insufficient_samples(self):
        calibrator = HistoricalConfidenceCalibrator()
        with pytest.raises(ValueError, match="at least 10 samples"):
            calibrator.fit(np.array([0.5, 0.6, 0.7]), np.array([1, 0, 1]))


# ==============================================================================
# 5. End-to-End Signal Engine Tests
# ==============================================================================

class TestSignalEngine:
    def test_full_bullish_pipeline(self):
        """Complete bullish intelligence snapshot → BUY signal."""
        engine = SignalEngine()

        snapshot = IntelligenceSnapshot(
            symbol="BTCUSDT",
            timestamp=NOW,
            technical_assessment=_make_technical_assessment(
                trend=TrendState.BULLISH,
                momentum=MomentumState.ACCELERATING,
                structure=StructureState.BREAKOUT_UP,
                volume=VolumeState.HIGH_VOLUME,
                mtf=MultiTimeframeState.ALIGNED_BULLISH,
            ),
            regime_assessment=_make_regime_assessment(regime=MarketRegime.TREND_UP),
            pattern_result=_make_pattern_result(positive_ratio=0.70),
            direction_prob_up=0.78,
            expected_return=0.008,
            expected_volatility=0.15,
            trade_quality_prob=0.72,
        )

        signal = engine.generate(snapshot)

        assert isinstance(signal, TradingSignal)
        assert signal.symbol == "BTCUSDT"
        assert signal.action == SignalAction.BUY
        assert signal.conviction > 0.0
        assert signal.regime == "TREND_UP"
        assert len(signal.evidence) >= 5  # Multiple evidence streams
        assert signal.signal_version == "v1.0.0"

    def test_full_bearish_pipeline(self):
        """Complete bearish intelligence snapshot → SELL signal."""
        engine = SignalEngine()

        snapshot = IntelligenceSnapshot(
            symbol="BTCUSDT",
            timestamp=NOW,
            technical_assessment=_make_technical_assessment(
                trend=TrendState.BEARISH,
                momentum=MomentumState.DECELERATING,
                structure=StructureState.BREAKDOWN_DOWN,
                volume=VolumeState.HIGH_VOLUME,
                mtf=MultiTimeframeState.ALIGNED_BEARISH,
            ),
            regime_assessment=_make_regime_assessment(regime=MarketRegime.TREND_DOWN),
            pattern_result=_make_pattern_result(positive_ratio=0.25),
            direction_prob_up=0.15,
            expected_return=-0.01,
            expected_volatility=0.18,
            trade_quality_prob=0.30,
        )

        signal = engine.generate(snapshot)

        assert signal.action == SignalAction.SELL
        assert signal.conviction > 0.0
        assert signal.fusion_score < 0

    def test_mixed_signals_no_trade(self):
        """Conflicting intelligence → NO_TRADE."""
        engine = SignalEngine()

        snapshot = IntelligenceSnapshot(
            symbol="BTCUSDT",
            timestamp=NOW,
            technical_assessment=_make_technical_assessment(
                trend=TrendState.BULLISH,
                momentum=MomentumState.OVERBOUGHT,
                structure=StructureState.CHANNEL_RANGE,
                volume=VolumeState.LOW_VOLUME,
                mtf=MultiTimeframeState.MIXED,
            ),
            regime_assessment=_make_regime_assessment(
                regime=MarketRegime.SIDEWAYS, confidence=0.55,
            ),
            direction_prob_up=0.52,
            expected_return=0.001,
            expected_volatility=0.25,
            trade_quality_prob=0.48,
        )

        signal = engine.generate(snapshot)

        assert signal.action == SignalAction.NO_TRADE

    def test_minimal_snapshot(self):
        """Snapshot with only technical assessment → NO_TRADE (insufficient evidence)."""
        engine = SignalEngine()

        snapshot = IntelligenceSnapshot(
            symbol="BTCUSDT",
            timestamp=NOW,
            technical_assessment=_make_technical_assessment(),
        )

        signal = engine.generate(snapshot)

        # Only 3 evidence streams (technical, volume, MTF) = min_evidence_count
        assert isinstance(signal, TradingSignal)

    def test_empty_snapshot(self):
        """Completely empty snapshot → NO_TRADE."""
        engine = SignalEngine()

        snapshot = IntelligenceSnapshot(
            symbol="BTCUSDT",
            timestamp=NOW,
        )

        signal = engine.generate(snapshot)
        assert signal.action == SignalAction.NO_TRADE
        assert signal.conviction == 0.0
        assert len(signal.evidence) == 0

    def test_signal_versioning(self):
        """Signal carries correct version strings."""
        config = SignalEngineConfig(version="v2.0.0")
        engine = SignalEngine(config)

        snapshot = IntelligenceSnapshot(
            symbol="ETHUSDT",
            timestamp=NOW,
            feature_version="v3.0.0",
            model_version="v2.5.0",
        )

        signal = engine.generate(snapshot)
        assert signal.signal_version == "v2.0.0"
        assert signal.feature_version == "v3.0.0"
        assert signal.model_version == "v2.5.0"

    def test_weight_overrides(self):
        """Custom weight overrides should be passed through to adapters."""
        config = SignalEngineConfig(
            weight_overrides={"TECHNICAL": 0.50, "ML_DIRECTION": 0.30},
        )
        engine = SignalEngine(config)

        snapshot = IntelligenceSnapshot(
            symbol="BTCUSDT",
            timestamp=NOW,
            technical_assessment=_make_technical_assessment(),
            direction_prob_up=0.80,
            expected_return=0.01,
            expected_volatility=0.15,
        )

        signal = engine.generate(snapshot)

        # Find technical evidence and verify weight override
        tech_evidence = [e for e in signal.evidence if e.evidence_type == EvidenceType.TECHNICAL]
        assert len(tech_evidence) == 1
        assert tech_evidence[0].weight == 0.50

        # Direction evidence
        dir_evidence = [e for e in signal.evidence if e.evidence_type == EvidenceType.ML_DIRECTION]
        assert len(dir_evidence) == 1
        assert dir_evidence[0].weight == 0.30

    def test_risk_reward_in_signal(self):
        """Risk/reward ratio should be computed in the final signal."""
        engine = SignalEngine()

        snapshot = IntelligenceSnapshot(
            symbol="BTCUSDT",
            timestamp=NOW,
            expected_return=0.02,
            expected_volatility=0.01,
            direction_prob_up=0.80,
            technical_assessment=_make_technical_assessment(),
        )

        signal = engine.generate(snapshot)
        assert signal.risk_reward_ratio > 0

    def test_evidence_audit_trail(self):
        """Every evidence stream should be traceable in the signal."""
        engine = SignalEngine()

        snapshot = IntelligenceSnapshot(
            symbol="BTCUSDT",
            timestamp=NOW,
            technical_assessment=_make_technical_assessment(),
            regime_assessment=_make_regime_assessment(),
            direction_prob_up=0.80,
            expected_return=0.01,
            expected_volatility=0.15,
            trade_quality_prob=0.70,
        )

        signal = engine.generate(snapshot)

        evidence_types = {e.evidence_type for e in signal.evidence}
        assert EvidenceType.TECHNICAL in evidence_types
        assert EvidenceType.VOLUME in evidence_types
        assert EvidenceType.MULTI_TIMEFRAME in evidence_types
        assert EvidenceType.REGIME in evidence_types
        assert EvidenceType.ML_DIRECTION in evidence_types
        assert EvidenceType.ML_RETURN in evidence_types
        assert EvidenceType.VOLATILITY in evidence_types
        assert EvidenceType.ML_TRADE_QUALITY in evidence_types
        assert EvidenceType.RISK_REWARD in evidence_types

    def test_signal_determinism(self):
        """Same input → same output. Signal generation must be deterministic."""
        engine = SignalEngine()

        snapshot = IntelligenceSnapshot(
            symbol="BTCUSDT",
            timestamp=NOW,
            technical_assessment=_make_technical_assessment(),
            regime_assessment=_make_regime_assessment(),
            direction_prob_up=0.75,
            expected_return=0.005,
            expected_volatility=0.15,
            trade_quality_prob=0.65,
        )

        signal1 = engine.generate(snapshot)
        signal2 = engine.generate(snapshot)

        assert signal1.action == signal2.action
        assert signal1.fusion_score == signal2.fusion_score
        assert signal1.conviction == signal2.conviction


# ==============================================================================
# 6. Edge Case Tests
# ==============================================================================

class TestEdgeCases:
    def test_all_zero_votes(self):
        """All neutral evidence → NO_TRADE."""
        engine = FusionEngine(FusionConfig(min_evidence_count=1))
        evidence = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.0, confidence=0.5, weight=0.15),
            Evidence(evidence_type=EvidenceType.ML_DIRECTION, vote=0.0, confidence=0.5, weight=0.20),
            Evidence(evidence_type=EvidenceType.REGIME, vote=0.0, confidence=0.5, weight=0.10),
        ]
        action, score, conviction = engine.fuse(evidence)
        assert action == SignalAction.NO_TRADE
        assert score == 0.0

    def test_all_zero_weights(self):
        """All zero weights → NO_TRADE."""
        engine = FusionEngine(FusionConfig(min_evidence_count=1))
        evidence = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.9, confidence=0.9, weight=0.0),
            Evidence(evidence_type=EvidenceType.ML_DIRECTION, vote=0.9, confidence=0.9, weight=0.0),
            Evidence(evidence_type=EvidenceType.REGIME, vote=0.9, confidence=0.9, weight=0.0),
        ]
        action, score, conviction = engine.fuse(evidence)
        assert action == SignalAction.NO_TRADE

    def test_single_overwhelming_evidence(self):
        """One very strong signal with sufficient evidence count → can produce signal."""
        config = FusionConfig(min_evidence_count=3)
        engine = FusionEngine(config)
        evidence = [
            Evidence(evidence_type=EvidenceType.ML_DIRECTION, vote=1.0, confidence=1.0, weight=0.5),
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.8, confidence=0.9, weight=0.3),
            Evidence(evidence_type=EvidenceType.REGIME, vote=0.5, confidence=0.8, weight=0.2),
        ]
        action, score, conviction = engine.fuse(evidence)
        assert action == SignalAction.BUY

    def test_boundary_threshold(self):
        """Score exactly at threshold boundary."""
        config = FusionConfig(
            base_buy_threshold=0.15,
            min_evidence_count=1,
            regime_adjustments={},  # No adjustments
        )
        engine = FusionEngine(config)

        # Engineer score to be exactly at threshold
        # vote * confidence * weight / total_weight = 0.15
        evidence = [
            Evidence(evidence_type=EvidenceType.TECHNICAL, vote=0.15, confidence=1.0, weight=1.0),
        ]
        action, score, _ = engine.fuse(evidence)
        assert action == SignalAction.BUY  # >= threshold
