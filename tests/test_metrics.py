"""Tests for MetricsCollector."""

import pytest
from src.hbr_apple_rag.metrics import MetricsCollector


class TestMetricsCollector:
    def test_start_and_stop(self):
        metrics = MetricsCollector()
        metrics.start()
        import time
        time.sleep(0.01)
        elapsed = metrics.stop()

        assert elapsed > 0
        assert metrics.response_time == elapsed
        assert metrics.has_measurement is True

    def test_stop_without_start_raises_error(self):
        metrics = MetricsCollector()
        with pytest.raises(RuntimeError):
            metrics.stop()

    def test_reset(self):
        metrics = MetricsCollector()
        metrics.start()
        metrics.stop()
        metrics.reset()

        assert metrics.response_time is None
        assert metrics.has_measurement is False