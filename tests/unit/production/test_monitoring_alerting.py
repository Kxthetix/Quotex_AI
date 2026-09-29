"""Unit tests for Monitoring, Metrics Registry, and Alerting (Sections 23, 24, 25)."""

import time

from production.monitoring.alerting import AlertManager, AlertSeverity
from production.monitoring.metrics import MetricsRegistry


def test_metrics_registry_counters_and_gauges():
    registry = MetricsRegistry()

    # Counter test
    registry.inc_counter("market_events_total", 5.0)
    assert registry.counters["market_events_total"].value == 5.0

    registry.inc_counter("market_events_total", 2.0)
    assert registry.counters["market_events_total"].value == 7.0

    # Gauge test
    registry.set_gauge("active_exposure", 25000.0)
    assert registry.gauges["active_exposure"].value == 25000.0

    registry.set_gauge("active_exposure", 18000.0)
    assert registry.gauges["active_exposure"].value == 18000.0


def test_metrics_registry_histogram_latency():
    registry = MetricsRegistry()

    with registry.time_block("processing_latency"):
        time.sleep(0.01)

    summary = registry.histograms["processing_latency"].summary()
    assert summary["count"] == 1.0
    assert summary["mean"] > 0.005

    snapshot = registry.collect()
    assert "counters" in snapshot
    assert "gauges" in snapshot
    assert "histograms" in snapshot


def test_alert_manager_severity_routing_and_cooldown():
    alert_mgr = AlertManager(cooldown_seconds=1.0)
    received_alerts = []

    alert_mgr.subscribe(lambda a: received_alerts.append(a))

    # Trigger critical alert
    alt1 = alert_mgr.trigger(
        severity=AlertSeverity.CRITICAL,
        component="MarketData",
        title="WebSocket Feed Dropped",
        description="No tick received for 20 seconds",
    )
    assert alt1 is not None
    assert len(received_alerts) == 1
    assert len(alert_mgr.get_active_critical_alerts()) == 1

    # Trigger identical alert within cooldown -> should be suppressed
    alt2 = alert_mgr.trigger(
        severity=AlertSeverity.CRITICAL,
        component="MarketData",
        title="WebSocket Feed Dropped",
        description="Duplicate alert",
    )
    assert alt2 is None
    assert len(received_alerts) == 1

    # Resolve active alert
    assert alert_mgr.resolve_alert(alt1.alert_id) is True
    assert len(alert_mgr.get_active_critical_alerts()) == 0
