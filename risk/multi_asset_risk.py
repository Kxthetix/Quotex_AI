"""Multi-Asset Portfolio Risk Engine (Section 44).

Evaluates cross-asset concentration, portfolio-wide aggregate exposure,
and asset correlation limits. Vetoes orders violating risk limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MultiAssetRiskAssessment:
    approved: bool
    symbol: str
    proposed_exposure: float
    total_portfolio_exposure: float
    portfolio_equity: float
    concentration_pct: float
    veto_reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class MultiAssetRiskEngine:
    """Manages cross-asset risk boundaries and concentration limits."""

    def __init__(
        self,
        max_single_asset_concentration_pct: float = 35.0,
        max_aggregate_leverage: float = 1.0,  # 1.0 = 100% max equity invested
        asset_correlation_threshold: float = 0.85,
    ) -> None:
        self.max_concentration_pct = max_single_asset_concentration_pct
        self.max_leverage = max_aggregate_leverage
        self.correlation_threshold = asset_correlation_threshold

    def evaluate_order(
        self,
        symbol: str,
        order_side: str,
        order_value: float,
        current_positions_value: dict[str, float],
        total_equity: float,
        pairwise_correlations: dict[tuple[str, str], float] | None = None,
    ) -> MultiAssetRiskAssessment:
        """Evaluate order against portfolio-wide multi-asset limits."""
        if total_equity <= 0:
            return MultiAssetRiskAssessment(
                approved=False,
                symbol=symbol,
                proposed_exposure=order_value,
                total_portfolio_exposure=0.0,
                portfolio_equity=total_equity,
                concentration_pct=100.0,
                veto_reason="Portfolio equity is non-positive or exhausted.",
            )

        clean_sym = symbol.upper()
        current_sym_val = current_positions_value.get(clean_sym, 0.0)

        # 1. Calculate proposed single-asset exposure
        if order_side.upper() == "BUY":
            new_sym_exposure = current_sym_val + order_value
        else:
            new_sym_exposure = max(0.0, current_sym_val - order_value)

        concentration_pct = (new_sym_exposure / total_equity) * 100.0

        # Check single-asset concentration limit
        if concentration_pct > self.max_concentration_pct:
            return MultiAssetRiskAssessment(
                approved=False,
                symbol=clean_sym,
                proposed_exposure=new_sym_exposure,
                total_portfolio_exposure=sum(current_positions_value.values()) + order_value,
                portfolio_equity=total_equity,
                concentration_pct=round(concentration_pct, 2),
                veto_reason=(
                    f"Concentration {concentration_pct:.1f}% exceeds maximum allowable "
                    f"{self.max_concentration_pct:.1f}% for {clean_sym}"
                ),
            )

        # 2. Check aggregate portfolio exposure
        other_positions_sum = sum(v for s, v in current_positions_value.items() if s != clean_sym)
        new_total_exposure = other_positions_sum + new_sym_exposure
        current_leverage = new_total_exposure / total_equity

        if current_leverage > self.max_leverage:
            return MultiAssetRiskAssessment(
                approved=False,
                symbol=clean_sym,
                proposed_exposure=new_sym_exposure,
                total_portfolio_exposure=new_total_exposure,
                portfolio_equity=total_equity,
                concentration_pct=round(concentration_pct, 2),
                veto_reason=(
                    f"Total portfolio leverage {current_leverage:.2f}x exceeds maximum "
                    f"{self.max_leverage:.2f}x"
                ),
            )

        # 3. Check pairwise correlation if provided
        if pairwise_correlations:
            for (s1, s2), corr in pairwise_correlations.items():
                if clean_sym in (s1, s2) and abs(corr) > self.correlation_threshold:
                    other_sym = s2 if clean_sym == s1 else s1
                    if other_positions_value := current_positions_value.get(other_sym, 0.0) > 0:
                        return MultiAssetRiskAssessment(
                            approved=False,
                            symbol=clean_sym,
                            proposed_exposure=new_sym_exposure,
                            total_portfolio_exposure=new_total_exposure,
                            portfolio_equity=total_equity,
                            concentration_pct=round(concentration_pct, 2),
                            veto_reason=(
                                f"High correlation ({corr:.2f}) between {clean_sym} and existing holding "
                                f"{other_sym} exceeds threshold {self.correlation_threshold}"
                            ),
                        )

        # Approved
        return MultiAssetRiskAssessment(
            approved=True,
            symbol=clean_sym,
            proposed_exposure=new_sym_exposure,
            total_portfolio_exposure=new_total_exposure,
            portfolio_equity=total_equity,
            concentration_pct=round(concentration_pct, 2),
            veto_reason=None,
        )
