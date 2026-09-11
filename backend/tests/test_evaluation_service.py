import pytest

from app.services.evaluation_service import classification_metrics, detection_rate, latency_summary


def test_metrics_are_calculated_from_predictions() -> None:
    metrics = classification_metrics(["A", "UNKNOWN", "A"], ["A", "A", "UNKNOWN"])
    assert metrics.accuracy == pytest.approx(1 / 3)
    assert metrics.far == pytest.approx(1.0)
    assert metrics.frr == pytest.approx(0.5)


def test_detection_and_latency_metrics() -> None:
    assert detection_rate(8, 10) == pytest.approx(0.8)
    assert latency_summary([10, 20, 30, 40])["median_ms"] == 25
