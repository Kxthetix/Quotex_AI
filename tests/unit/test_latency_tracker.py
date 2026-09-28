"""Unit tests for latency percentiles and health report."""

from monitoring.health import LatencyTracker, SystemHealthMonitor


def test_latency_tracker_percentiles():
    tracker = LatencyTracker(window_size=100)

    # Empty
    metrics_empty = tracker.get_metrics()
    assert metrics_empty.sample_count == 0
    assert metrics_empty.p50_ms == 0.0

    # Add 100 samples from 1 to 100 ms
    for i in range(1, 101):
        tracker.record_latency_ms(float(i))

    metrics = tracker.get_metrics()
    assert metrics.sample_count == 100
    assert metrics.min_ms == 1.0
    assert metrics.max_ms == 100.0
    assert abs(metrics.p50_ms - 50.5) < 1.0
    assert abs(metrics.p95_ms - 95.05) < 1.0
    assert abs(metrics.p99_ms - 99.01) < 1.0


def test_system_health_monitor_healthy():
    tracker = LatencyTracker()
    tracker.record_latency_ms(15.0)
    monitor = SystemHealthMonitor(latency_tracker=tracker)

    report = monitor.generate_report(is_connected=True, is_breaker_tripped=False)
    assert report.is_healthy
    assert report.can_trade
    assert report.status == "HEALTHY"
    assert report.latency.p50_ms == 15.0


def test_system_health_monitor_breaker_tripped():
    tracker = LatencyTracker()
    monitor = SystemHealthMonitor(latency_tracker=tracker)

    report = monitor.generate_report(
        is_connected=True,
        is_breaker_tripped=True,
        breaker_reason="Corrupt ticks detected",
    )
    assert not report.can_trade
    assert report.breaker_tripped
    assert any("circuit breaker is tripped" in r for r in report.reasons)
