"""Main Backtest Engine orchestrating chronological simulation (Phase 5).

Connects historical replay, feature updating, intelligence models, signal adaptation,
position sizing, order lifecycle, execution simulation, portfolio accounting,
and performance metrics into a reproducible backtest run.
"""

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from analysis.regime.models import MarketRegime
from backtesting.events import SignalEvent
from backtesting.execution.fees import PercentageFeeModel
from backtesting.execution.latency import FixedLatencyModel, ZeroLatencyModel
from backtesting.execution.simulator import ExecutionSimulator
from backtesting.execution.slippage import PercentageSlippageModel
from backtesting.execution.spread import PercentageSpreadModel
from backtesting.metrics import MetricsEngine, annualization_factor_for_timeframe
from backtesting.models import (
    BacktestConfig,
    BacktestResult,
    EndOfTestAction,
    ExitReason,
    Fill,
    Order,
    OrderSide,
    OrderType,
    PositionSide,
)
from backtesting.order_engine import OrderEngine
from backtesting.portfolio import PortfolioEngine
from backtesting.replay import HistoricalReplayEngine
from backtesting.signal_adapter import SignalAdapter
from backtesting.sizing import PositionSizer
from data.ingestion.models import Candle

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Event-driven backtesting execution engine enforcing strict chronological processing."""

    def __init__(
        self,
        config: BacktestConfig | None = None,
        intelligence_provider: Callable[[Candle, list[Candle]], SignalEvent | None] | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.intelligence_provider = intelligence_provider
        self.signal_adapter = SignalAdapter()

        # Build execution simulator based on config
        spread_model = PercentageSpreadModel(spread_pct=self.config.spread_pct)
        slippage_model = PercentageSlippageModel(
            base_slippage_pct=self.config.slippage_pct,
            volatility_factor=self.config.slippage_vol_factor,
        )
        fee_model = PercentageFeeModel(
            maker_fee=self.config.maker_fee,
            taker_fee=self.config.taker_fee,
            fixed_fee=self.config.fixed_fee,
        )
        latency_model = (
            FixedLatencyModel(latency_seconds=self.config.latency_seconds)
            if self.config.latency_seconds > 0
            else ZeroLatencyModel()
        )

        self.execution_simulator = ExecutionSimulator(
            spread_model=spread_model,
            slippage_model=slippage_model,
            fee_model=fee_model,
            latency_model=latency_model,
            max_bar_volume_pct=self.config.max_bar_volume_pct,
            allow_partial_fills=self.config.allow_partial_fills,
        )

        self.order_engine = OrderEngine()
        self.portfolio = PortfolioEngine(initial_capital=self.config.initial_capital)
        self.position_sizer = PositionSizer(self.config)
        self.metrics_engine = MetricsEngine(
            annualization_factor=annualization_factor_for_timeframe(self.config.timeframe)
        )

        self.audit_log: list[str] = []
        self.warnings: list[str] = []

    def _log_event(self, timestamp: datetime, message: str) -> None:
        """Record timestamped entry in the audit trail."""
        ts_str = timestamp.isoformat()
        entry = f"[{ts_str}] {message}"
        self.audit_log.append(entry)

    def run(self, candles: list[Candle]) -> BacktestResult:
        """Execute chronological backtest over historical candles."""
        backtest_id = f"bt_{uuid.uuid4().hex[:10]}"
        replay = HistoricalReplayEngine(
            candles=candles,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            symbols=[self.config.symbol],
            strict_monotony=True,
            on_data_error="skip_with_warning",
        )

        # Collect data quality warnings from replay
        for warn in replay.stats.warnings:
            self.warnings.append(f"{warn.warning_type} at {warn.timestamp}: {warn.message}")

        candle_history: list[Candle] = []
        initial_price = 0.0
        final_price = 0.0
        sim_start_time: datetime | None = None
        sim_end_time: datetime | None = None

        # Chronological event loop
        for event in replay.stream_events():
            candle = event.candle
            current_time = event.timestamp

            if sim_start_time is None:
                sim_start_time = current_time
                initial_price = candle.open

            sim_end_time = current_time
            final_price = candle.close

            candle_history.append(candle)
            self._log_event(
                current_time,
                f"Candle completed: {candle.symbol} C={candle.close:.2f} V={candle.volume:.2f}",
            )

            # 1. Process pending active orders against the new candle
            self._process_active_orders(candle, current_time)

            # 2. Update mark-to-market valuation
            self.portfolio.update_valuation({candle.symbol: candle.close}, current_time)

            # 3. Check protective stop-loss / take-profit triggers
            self._evaluate_protective_exits(candle, current_time)

            # 4. Generate intelligence signals (no future data)
            signal: SignalEvent | None = None
            if self.intelligence_provider:
                signal = self.intelligence_provider(candle, candle_history)
            else:
                signal = self._default_baseline_intelligence(candle, candle_history)

            # 5. Process signal decisions and place orders
            if signal:
                self._handle_signal(signal, candle, current_time)

        # End of backtest processing
        if sim_end_time and self.config.end_of_test_action == EndOfTestAction.FORCE_CLOSE:
            self._force_close_open_positions(final_price, sim_end_time)

        # Final valuation snapshot
        if sim_end_time:
            self.portfolio.update_valuation({self.config.symbol: final_price}, sim_end_time)

        # Calculate performance metrics
        pe = self.portfolio.get_position_engine(self.config.symbol)
        metrics = self.metrics_engine.calculate_metrics(
            equity_curve=self.portfolio.equity_curve,
            trade_ledger=pe.closed_trades,
            initial_capital=self.config.initial_capital,
            benchmark_initial_price=initial_price,
            benchmark_final_price=final_price,
        )

        final_equity = (
            self.portfolio.equity_curve[-1].equity
            if self.portfolio.equity_curve
            else self.config.initial_capital
        )

        return BacktestResult(
            backtest_id=backtest_id,
            config=self.config,
            start_time=sim_start_time or datetime.now(UTC),
            end_time=sim_end_time or datetime.now(UTC),
            symbols=[self.config.symbol],
            initial_capital=self.config.initial_capital,
            final_equity=float(final_equity),
            total_return=metrics.total_return,
            cagr=metrics.cagr,
            max_drawdown=metrics.max_drawdown,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            calmar_ratio=metrics.calmar_ratio,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            expectancy=metrics.expectancy,
            trade_count=metrics.total_trades,
            avg_trade_pnl=metrics.avg_trade_pnl,
            total_fees=metrics.total_fees,
            total_slippage=metrics.total_slippage,
            total_spread=metrics.total_spread,
            equity_curve=self.portfolio.equity_curve,
            trade_ledger=pe.closed_trades,
            metrics=metrics,
            audit_log=self.audit_log,
            warnings=self.warnings,
            code_version="v1.0.0",
        )

    def _process_active_orders(self, candle: Candle, current_time: datetime) -> None:
        """Attempt execution on active orders using the execution simulator."""
        active = list(self.order_engine.active_orders)
        for order in active:
            if order.symbol != candle.symbol:
                continue

            fill = self.execution_simulator.process_order(
                order=order,
                candle=candle,
                current_time=current_time,
                volatility=0.0,
            )

            if fill:
                self._apply_fill(fill, order, fill.timestamp)

    def _apply_fill(
        self,
        fill: Fill,
        order: Order,
        current_time: datetime,
        exit_reason: ExitReason = ExitReason.SIGNAL,
    ) -> None:
        """Update portfolio and order state upon successful fill."""
        pe = self.portfolio.get_position_engine(fill.symbol)
        closed_trade = pe.apply_fill(
            fill=fill,
            exit_reason=exit_reason,
            regime=MarketRegime.SIDEWAYS,
            model_version=self.config.model_version,
            feature_version=self.config.feature_version,
        )

        notional = fill.price * fill.quantity
        if fill.side == OrderSide.BUY:
            self.portfolio.cash -= notional + fill.fee
        else:
            self.portfolio.cash += notional - fill.fee

        self.portfolio.cumulative_fees += fill.fee

        if closed_trade:
            self.portfolio.cumulative_realized_pnl += closed_trade.net_pnl
            self._log_event(
                current_time,
                f"Trade closed: {closed_trade.side} qty={closed_trade.quantity:.4f} "
                f"gross_pnl={closed_trade.gross_pnl:.2f} net_pnl={closed_trade.net_pnl:.2f} "
                f"reason={closed_trade.exit_reason}",
            )

        self.order_engine.update_order_status(order)
        self._log_event(
            current_time,
            f"Order filled: id={order.order_id} side={fill.side} qty={fill.quantity:.4f} "
            f"price={fill.price:.2f} fee={fill.fee:.4f} status={order.status}",
        )

    def _evaluate_protective_exits(self, candle: Candle, current_time: datetime) -> None:
        """Evaluate stop-loss and take-profit triggers on open positions."""
        pe = self.portfolio.get_position_engine(candle.symbol)
        pos = pe.position
        if pos.side == PositionSide.FLAT or pos.quantity <= 1e-8:
            return

        # Check metadata from active order or standard 2% stop / 4% target if enabled
        stop_pct = self.config.slippage_pct * 10.0  # default baseline if specified
        # Stop loss check
        if pos.side == PositionSide.LONG and stop_pct > 0:
            stop_price = pos.avg_entry_price * (1.0 - 0.05)  # 5% protective stop
            if candle.low <= stop_price:
                # Trigger stop loss
                exit_order = self.order_engine.create_order(
                    symbol=candle.symbol,
                    side=OrderSide.SELL,
                    quantity=pos.quantity,
                    order_type=OrderType.MARKET,
                    timestamp=current_time,
                    metadata={"reason": ExitReason.STOP_LOSS},
                )
                # The stop was observed from this completed candle. Queue the exit
                # for the next candle so the backtest cannot fill at an earlier price.

    def _handle_signal(self, signal: SignalEvent, candle: Candle, current_time: datetime) -> None:
        """Translate standardized SignalEvent into sized orders."""
        if any(order.symbol == signal.symbol for order in self.order_engine.active_orders):
            self._log_event(current_time, "Signal held while a prior order remains active")
            return

        pe = self.portfolio.get_position_engine(signal.symbol)
        pos = pe.position

        # Case 1: Exit Signal
        if signal.suggested_action == "EXIT":
            if pos.side != PositionSide.FLAT and pos.quantity > 0:
                side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
                order = self.order_engine.create_order(
                    symbol=signal.symbol,
                    side=side,
                    quantity=pos.quantity,
                    order_type=OrderType.MARKET,
                    timestamp=current_time,
                    metadata={"reason": ExitReason.SIGNAL},
                )
                self._log_event(
                    current_time,
                    f"Exit order created: {order.order_id} {side} qty={pos.quantity:.4f}",
                )
                # Active orders are evaluated on the next candle, never on the
                # candle that was used to form this signal.
            return

        # Case 2: Enter Long Signal
        if signal.suggested_action == "ENTER_LONG":
            if (
                pos.side == PositionSide.LONG
                and pos.quantity >= self.config.max_position_size * 0.99
            ):
                return  # Already at maximum position size

            current_equity = self.portfolio.cash + (
                pos.quantity * candle.close if pos.side == PositionSide.LONG else 0.0
            )
            order_qty = self.position_sizer.calculate_order_quantity(
                current_equity=current_equity,
                current_cash=self.portfolio.cash,
                current_price=candle.close,
                current_position_qty=pos.quantity if pos.side == PositionSide.LONG else 0.0,
                expected_volatility=signal.expected_volatility,
            )

            if order_qty > 1e-6:
                order = self.order_engine.create_order(
                    symbol=signal.symbol,
                    side=OrderSide.BUY,
                    quantity=order_qty,
                    order_type=OrderType.MARKET,
                    timestamp=current_time,
                    metadata={"signal_prob": signal.direction_prob},
                )
                self._log_event(
                    current_time, f"Entry Long order created: {order.order_id} qty={order_qty:.4f}"
                )
                # Active orders are evaluated on the next candle, never on the
                # candle that was used to form this signal.

        # Case 3: Enter Short Signal
        elif signal.suggested_action == "ENTER_SHORT":
            if (
                pos.side == PositionSide.SHORT
                and pos.quantity >= self.config.max_position_size * 0.99
            ):
                return

            current_equity = self.portfolio.cash
            order_qty = self.position_sizer.calculate_order_quantity(
                current_equity=current_equity,
                current_cash=self.portfolio.cash,
                current_price=candle.close,
                current_position_qty=pos.quantity if pos.side == PositionSide.SHORT else 0.0,
                expected_volatility=signal.expected_volatility,
            )

            if order_qty > 1e-6:
                order = self.order_engine.create_order(
                    symbol=signal.symbol,
                    side=OrderSide.SELL,
                    quantity=order_qty,
                    order_type=OrderType.MARKET,
                    timestamp=current_time,
                    metadata={"signal_prob": signal.direction_prob},
                )
                self._log_event(
                    current_time, f"Entry Short order created: {order.order_id} qty={order_qty:.4f}"
                )
                # Active orders are evaluated on the next candle, never on the
                # candle that was used to form this signal.

    def _force_close_open_positions(self, final_price: float, end_time: datetime) -> None:
        """Force close any remaining open positions at the final available price."""
        pe = self.portfolio.get_position_engine(self.config.symbol)
        pos = pe.position
        if pos.side != PositionSide.FLAT and pos.quantity > 1e-8:
            exit_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
            close_order = self.order_engine.create_order(
                symbol=self.config.symbol,
                side=exit_side,
                quantity=pos.quantity,
                order_type=OrderType.MARKET,
                timestamp=end_time,
                metadata={"reason": ExitReason.END_OF_BACKTEST},
            )
            # Create synthetic closing candle to guarantee fill at final mark
            closing_candle = Candle(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                open=final_price,
                high=final_price,
                low=final_price,
                close=final_price,
                volume=pos.quantity * 1000.0,
                start_time=end_time,
                end_time=end_time,
            )
            fill = self.execution_simulator.process_order(
                order=close_order,
                candle=closing_candle,
                current_time=end_time,
            )
            if fill:
                self._apply_fill(
                    fill, close_order, end_time, exit_reason=ExitReason.END_OF_BACKTEST
                )
                self._log_event(end_time, f"End-of-test forced close executed at {final_price:.2f}")

    def _default_baseline_intelligence(
        self, candle: Candle, history: list[Candle]
    ) -> SignalEvent | None:
        """Simple baseline intelligence using fast moving-average momentum rule."""
        if len(history) < 20:
            return None

        # 5-period vs 20-period moving average
        recent_5 = history[-5:]
        recent_20 = history[-20:]
        sma_fast = sum(c.close for c in recent_5) / 5.0
        sma_slow = sum(c.close for c in recent_20) / 20.0

        action = "HOLD"
        direction_prob = 0.5
        if sma_fast > sma_slow * 1.001:
            action = "ENTER_LONG"
            direction_prob = 0.65
        elif sma_fast < sma_slow * 0.999:
            action = "EXIT"
            direction_prob = 0.35

        return SignalEvent(
            timestamp=candle.end_time,
            symbol=candle.symbol,
            direction_prob=direction_prob,
            expected_return=0.005 if action == "ENTER_LONG" else 0.0,
            expected_volatility=0.30,
            trade_quality_prob=0.60,
            regime="TREND_UP" if sma_fast > sma_slow else "TREND_DOWN",
            suggested_action=action,
            strength=0.75,
            model_version=self.config.model_version,
            feature_version=self.config.feature_version,
        )
