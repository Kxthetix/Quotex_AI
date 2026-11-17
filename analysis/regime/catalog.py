"""Historical Market Regime Catalog and Presets.

Defines curated macro and micro historical market regimes across crypto history,
including precise time bounds, descriptive narratives, expected dominant regimes,
and characteristics for stress-testing and training algorithmic trading models.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from analysis.regime.models import MarketRegime


class RegimeCategory(StrEnum):
    """Broad classification of market dynamics."""

    BULL_EXPANSION = "BULL_EXPANSION"
    BEAR_CAPITULATION = "BEAR_CAPITULATION"
    LOW_VOL_CONSOLIDATION = "LOW_VOL_CONSOLIDATION"
    HIGH_VOL_SHOCK = "HIGH_VOL_SHOCK"
    RANGE_CHOP = "RANGE_CHOP"
    MULTI_YEAR_CYCLE = "MULTI_YEAR_CYCLE"


@dataclass(frozen=True)
class RegimePreset:
    """Pre-configured historical market regime window."""

    preset_id: str
    name: str
    category: RegimeCategory
    description: str
    start: datetime
    end: datetime
    default_symbol: str = "BTCUSDT"
    expected_primary_regimes: tuple[MarketRegime, ...] = ()

    @property
    def duration_days(self) -> int:
        return (self.end - self.start).days


REGIME_PRESETS: dict[str, RegimePreset] = {
    "bull_parabolic_2020_2021": RegimePreset(
        preset_id="bull_parabolic_2020_2021",
        name="2020-2021 Secular Bull Run / Parabolic Expansion",
        category=RegimeCategory.BULL_EXPANSION,
        description="Secular post-halving bull market driven by institutional inflows. BTC rallied from ~$10.5k to ~$64.8k (+517%).",
        start=datetime(2020, 10, 1, 0, 0, tzinfo=UTC),
        end=datetime(2021, 4, 15, 0, 0, tzinfo=UTC),
        expected_primary_regimes=(MarketRegime.TREND_UP, MarketRegime.BREAKOUT),
    ),
    "covid_crash_2020": RegimePreset(
        preset_id="covid_crash_2020",
        name="March 2020 Liquidity Shock / COVID Flash Crash",
        category=RegimeCategory.HIGH_VOL_SHOCK,
        description="Extreme high volatility black swan event with massive liquidations. BTC dropped ~50% in 48h ($9k to $3.8k).",
        start=datetime(2020, 3, 1, 0, 0, tzinfo=UTC),
        end=datetime(2020, 3, 31, 0, 0, tzinfo=UTC),
        expected_primary_regimes=(MarketRegime.HIGH_VOLATILITY, MarketRegime.TREND_DOWN, MarketRegime.REVERSAL),
    ),
    "post_may_crash_range_2021": RegimePreset(
        preset_id="post_may_crash_range_2021",
        name="Mid-2021 Wyckoff Re-accumulation Range",
        category=RegimeCategory.RANGE_CHOP,
        description="Wide-range consolidation following May liquidations and China mining ban, chopping between $30k and $42k.",
        start=datetime(2021, 5, 19, 0, 0, tzinfo=UTC),
        end=datetime(2021, 7, 21, 0, 0, tzinfo=UTC),
        expected_primary_regimes=(MarketRegime.CONSOLIDATION, MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOLATILITY),
    ),
    "bear_market_2022": RegimePreset(
        preset_id="bear_market_2022",
        name="2022 Macro Crypto Winter & Contagion",
        category=RegimeCategory.BEAR_CAPITULATION,
        description="Protracted macro bear market featuring Terra/Luna, 3AC, Celsius, and FTX collapses. BTC fell from $69k to $15.5k (-77%).",
        start=datetime(2021, 11, 15, 0, 0, tzinfo=UTC),
        end=datetime(2022, 11, 30, 0, 0, tzinfo=UTC),
        expected_primary_regimes=(MarketRegime.TREND_DOWN, MarketRegime.HIGH_VOLATILITY, MarketRegime.BREAKOUT),
    ),
    "low_vol_consolidation_2023": RegimePreset(
        preset_id="low_vol_consolidation_2023",
        name="Summer 2023 Volatility Compression",
        category=RegimeCategory.LOW_VOL_CONSOLIDATION,
        description="Historical volatility crush and compression where BTC chopped in a tight $25k-$31k range for nearly 4 months.",
        start=datetime(2023, 6, 25, 0, 0, tzinfo=UTC),
        end=datetime(2023, 10, 15, 0, 0, tzinfo=UTC),
        expected_primary_regimes=(MarketRegime.LOW_VOLATILITY, MarketRegime.CONSOLIDATION, MarketRegime.SIDEWAYS),
    ),
    "etf_bull_expansion_2023_2024": RegimePreset(
        preset_id="etf_bull_expansion_2023_2024",
        name="2023-2024 Spot ETF Approval Bull Rally",
        category=RegimeCategory.BULL_EXPANSION,
        description="High momentum breakout from $27k to new all-time high of $73.7k (+173%) driven by spot ETF inflows.",
        start=datetime(2023, 10, 16, 0, 0, tzinfo=UTC),
        end=datetime(2024, 3, 15, 0, 0, tzinfo=UTC),
        expected_primary_regimes=(MarketRegime.TREND_UP, MarketRegime.BREAKOUT),
    ),
    "post_halving_range_2024": RegimePreset(
        preset_id="post_halving_range_2024",
        name="Summer 2024 Post-Halving Re-accumulation Chop",
        category=RegimeCategory.RANGE_CHOP,
        description="5-month sideways consolidation range between $54,000 and $71,000 following the 2024 halving.",
        start=datetime(2024, 4, 1, 0, 0, tzinfo=UTC),
        end=datetime(2024, 9, 1, 0, 0, tzinfo=UTC),
        expected_primary_regimes=(MarketRegime.SIDEWAYS, MarketRegime.CONSOLIDATION, MarketRegime.REVERSAL),
    ),
    "multi_year_cycle_2021_2024": RegimePreset(
        preset_id="multi_year_cycle_2021_2024",
        name="3-Year Macro Market Cycle (2021-2024)",
        category=RegimeCategory.MULTI_YEAR_CYCLE,
        description="Continuous 3-year multi-regime dataset capturing bull mania, deep bear market, base accumulation, and pre-halving recovery.",
        start=datetime(2021, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        expected_primary_regimes=(
            MarketRegime.TREND_UP,
            MarketRegime.TREND_DOWN,
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.LOW_VOLATILITY,
            MarketRegime.BREAKOUT,
            MarketRegime.CONSOLIDATION,
        ),
    ),
    "multi_year_cycle_2022_2025": RegimePreset(
        preset_id="multi_year_cycle_2022_2025",
        name="3-Year Macro Market Cycle (2022-2025)",
        category=RegimeCategory.MULTI_YEAR_CYCLE,
        description="Continuous 3-year dataset encompassing bear market capitulation, accumulation bottom, and the 2024 ETF expansion cycle.",
        start=datetime(2022, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        expected_primary_regimes=(
            MarketRegime.TREND_UP,
            MarketRegime.TREND_DOWN,
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.LOW_VOLATILITY,
            MarketRegime.BREAKOUT,
            MarketRegime.CONSOLIDATION,
        ),
    ),
}


def get_regime_preset(preset_id: str) -> RegimePreset:
    """Retrieve a preset by identifier, raising KeyError if not found."""
    clean_id = preset_id.lower().strip()
    if clean_id not in REGIME_PRESETS:
        available = ", ".join(REGIME_PRESETS.keys())
        raise KeyError(f"Unknown regime preset '{preset_id}'. Available presets: {available}")
    return REGIME_PRESETS[clean_id]


def list_regime_presets() -> list[RegimePreset]:
    """Return all predefined regime presets."""
    return list(REGIME_PRESETS.values())
