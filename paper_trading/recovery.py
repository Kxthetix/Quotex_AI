"""Restart Recovery Engine for Phase 7 Paper Trading.

Enables seamless stop and restart of paper trading sessions:
- Snapshotting & state persistence
- Restoring Paper Portfolio, open positions, and cash balance
- Restoring Prediction Ledger and pending prediction evaluation queues
- Restoring Paper Trade Ledger
- Enforcing idempotent identifiers to strictly prevent duplicate predictions,
  orders, fills, or trades upon restart.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from monitoring.logging import get_logger
from paper_trading.execution import PaperExecutionEngine
from paper_trading.models import (
    ExitReason,
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperOrderType,
    PaperPortfolio,
    PaperPosition,
    PaperTradeRecord,
    PaperTradingSession,
    PredictionRecord,
    PredictionStatus,
    SessionStatus,
)
from paper_trading.portfolio import PaperPortfolioManager
from paper_trading.prediction_ledger import PredictionLedger

logger = get_logger(__name__)


class SessionRecoveryEngine:
    """Manages snapshotting, recovery, and idempotent deduplication across restarts."""

    def __init__(self, checkpoint_dir: Path | str = "data/paper_trading/checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        session: PaperTradingSession,
        portfolio_mgr: PaperPortfolioManager,
        execution_engine: PaperExecutionEngine,
        prediction_ledger: PredictionLedger,
    ) -> Path:
        """Create a complete deterministic checkpoint file for the session."""
        checkpoint_file = self.checkpoint_dir / f"{session.session_id}_checkpoint.json"

        data: dict[str, Any] = {
            "session": session.to_dict(),
            "portfolio": portfolio_mgr.portfolio.to_dict(),
            "open_positions": {
                sym: pos.to_dict() for sym, pos in portfolio_mgr.portfolio.positions.items()
            },
            "trade_ledger": [t.to_dict() for t in portfolio_mgr.get_trade_ledger()],
            "orders": [o.to_dict() for o in execution_engine.get_all_orders()],
            "predictions": [p.to_dict() for p in prediction_ledger.get_all_predictions()],
            "checkpoint_time": datetime.now(UTC).isoformat(),
        }

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("session_checkpoint_saved", session_id=session.session_id, path=str(checkpoint_file))
        return checkpoint_file

    def load_checkpoint(self, checkpoint_file: Path | str) -> dict[str, Any]:
        """Load raw checkpoint data from file."""
        file_path = Path(checkpoint_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    def restore_session(
        self,
        checkpoint_file: Path | str,
        portfolio_mgr: PaperPortfolioManager,
        execution_engine: PaperExecutionEngine,
        prediction_ledger: PredictionLedger,
    ) -> PaperTradingSession:
        """Restore all components from a saved checkpoint into active runtime objects."""
        data = self.load_checkpoint(checkpoint_file)

        # 1. Restore Session
        s_data = data["session"]
        start_t = datetime.fromisoformat(s_data["start_time"]) if s_data.get("start_time") else datetime.now(UTC)
        session = PaperTradingSession(
            session_id=s_data["session_id"],
            start_time=start_t,
            configuration_version=s_data.get("configuration_version", "v1.0.0"),
            model_versions=s_data.get("model_versions", {}),
            feature_version=s_data.get("feature_version", "v1.0.0"),
            signal_version=s_data.get("signal_version", "v1.0.0"),
            initial_capital=float(s_data.get("initial_capital", 10000.0)),
            status=SessionStatus.RUNNING,
            symbols=s_data.get("symbols", []),
            metadata=s_data.get("metadata", {}),
        )

        # 2. Restore Portfolio
        p_data = data["portfolio"]
        portfolio_mgr.portfolio.initial_capital = float(p_data["initial_capital"])
        portfolio_mgr.portfolio.cash = float(p_data["cash"])
        portfolio_mgr.portfolio.peak_equity = float(p_data.get("peak_equity", p_data["cash"]))
        portfolio_mgr.portfolio.total_fees = float(p_data.get("total_fees", 0.0))
        portfolio_mgr.portfolio.total_slippage = float(p_data.get("total_slippage", 0.0))
        portfolio_mgr.portfolio.cumulative_realized_pnl = float(p_data.get("realized_pnl", 0.0))

        # Restore Positions
        portfolio_mgr.portfolio.positions.clear()
        for sym, pos_data in data.get("open_positions", {}).items():
            pos = PaperPosition(
                symbol=pos_data["symbol"],
                side=pos_data["side"],
                quantity=float(pos_data["quantity"]),
                avg_entry_price=float(pos_data["avg_entry_price"]),
                current_price=float(pos_data["current_price"]),
                realized_pnl=float(pos_data.get("realized_pnl", 0.0)),
                cumulative_fees=float(pos_data.get("cumulative_fees", 0.0)),
            )
            portfolio_mgr.portfolio.positions[sym] = pos

        # 3. Restore Trade Ledger (idempotent, avoids duplicate trades)
        existing_trade_ids = {t.trade_id for t in portfolio_mgr.get_trade_ledger()}
        for t_data in data.get("trade_ledger", []):
            tid = t_data["trade_id"]
            if tid in existing_trade_ids:
                continue
            entry_t = datetime.fromisoformat(t_data["entry_time"]) if t_data.get("entry_time") else datetime.now(UTC)
            exit_t = datetime.fromisoformat(t_data["exit_time"]) if t_data.get("exit_time") else datetime.now(UTC)
            trade = PaperTradeRecord(
                trade_id=tid,
                signal_id=t_data.get("signal_id", ""),
                order_id=t_data.get("order_id", ""),
                symbol=t_data["symbol"],
                side=t_data["side"],
                entry_price=float(t_data["entry_price"]),
                exit_price=float(t_data["exit_price"]),
                quantity=float(t_data["quantity"]),
                fees=float(t_data.get("fees", 0.0)),
                slippage=float(t_data.get("slippage", 0.0)),
                pnl=float(t_data.get("pnl", 0.0)),
                return_pct=float(t_data.get("return_pct", 0.0)),
                entry_time=entry_t,
                exit_time=exit_t,
                holding_period_seconds=float(t_data.get("holding_period_seconds", 0.0)),
                exit_reason=ExitReason(t_data.get("exit_reason", "SIGNAL")),
                regime=t_data.get("regime", "SIDEWAYS"),
                model_version=t_data.get("model_version", "v1.0.0"),
            )
            portfolio_mgr.trade_ledger.append(trade)
            existing_trade_ids.add(tid)

        # 4. Restore Prediction Ledger (idempotent, restores pending evaluations)
        existing_pred_ids = {p.prediction_id for p in prediction_ledger.get_all_predictions()}
        for p_data in data.get("predictions", []):
            pid = p_data["prediction_id"]
            if pid in existing_pred_ids:
                continue
            p_time = datetime.fromisoformat(p_data["timestamp"]) if p_data.get("timestamp") else datetime.now(UTC)
            eval_time = datetime.fromisoformat(p_data["evaluation_time"]) if p_data.get("evaluation_time") else None
            pred = PredictionRecord(
                prediction_id=pid,
                symbol=p_data["symbol"],
                timestamp=p_time,
                timeframe=p_data.get("timeframe", "1m"),
                model_version=p_data.get("model_version", "v1.0.0"),
                feature_version=p_data.get("feature_version", "v1.0.0"),
                signal_version=p_data.get("signal_version", "v1.0.0"),
                signal_action=p_data.get("signal_action", "NO_TRADE"),
                confidence=float(p_data.get("confidence", 0.0)),
                regime=p_data.get("regime", "SIDEWAYS"),
                setup=p_data.get("setup", "STANDARD"),
                expected_return=float(p_data.get("expected_return", 0.0)),
                expected_volatility=float(p_data.get("expected_volatility", 0.0)),
                expected_duration_bars=int(p_data.get("expected_duration_bars", 15)),
                entry_price=float(p_data.get("entry_price", 0.0)),
                stop_price=float(p_data.get("stop_price", 0.0)),
                target_price=float(p_data.get("target_price", 0.0)),
                evaluation_horizon_seconds=float(p_data.get("evaluation_horizon_seconds", 900.0)),
                status=PredictionStatus(p_data.get("status", "PENDING")),
                evaluation_time=eval_time,
                realized_return=float(p_data["realized_return"]) if p_data.get("realized_return") is not None else None,
                mfe=float(p_data["mfe"]) if p_data.get("mfe") is not None else None,
                mae=float(p_data["mae"]) if p_data.get("mae") is not None else None,
                actual_outcome=p_data.get("actual_outcome"),
                prediction_error=float(p_data["prediction_error"]) if p_data.get("prediction_error") is not None else None,
                metadata=p_data.get("metadata", {}),
            )
            prediction_ledger.record_prediction(pred)
            existing_pred_ids.add(pid)

        # 5. Restore Orders
        for o_data in data.get("orders", []):
            oid = o_data["order_id"]
            if oid not in execution_engine._orders:
                c_time = datetime.fromisoformat(o_data["creation_time"]) if o_data.get("creation_time") else datetime.now(UTC)
                order = PaperOrder(
                    order_id=oid,
                    signal_id=o_data.get("signal_id", ""),
                    symbol=o_data["symbol"],
                    side=PaperOrderSide(o_data["side"]),
                    requested_quantity=float(o_data["requested_quantity"]),
                    order_type=PaperOrderType(o_data["order_type"]),
                    requested_price=float(o_data.get("requested_price", 0.0)),
                    limit_price=float(o_data["limit_price"]) if o_data.get("limit_price") is not None else None,
                    creation_time=c_time,
                    status=PaperOrderStatus(o_data.get("status", "CREATED")),
                    filled_quantity=float(o_data.get("filled_quantity", 0.0)),
                    avg_fill_price=float(o_data.get("avg_fill_price", 0.0)),
                    cumulative_fees=float(o_data.get("cumulative_fees", 0.0)),
                    cumulative_slippage=float(o_data.get("cumulative_slippage", 0.0)),
                )
                execution_engine._orders[oid] = order

        logger.info(
            "session_successfully_restored",
            session_id=session.session_id,
            trades_restored=len(portfolio_mgr.get_trade_ledger()),
            predictions_restored=len(prediction_ledger.get_all_predictions()),
            open_positions=len(portfolio_mgr.portfolio.positions),
        )
        return session
