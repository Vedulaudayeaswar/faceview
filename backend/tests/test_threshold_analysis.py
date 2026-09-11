import pytest

from app.services.evaluation_service import threshold_analysis


def test_threshold_analysis_uses_real_scores() -> None:
    rows = threshold_analysis([0.9, 0.8], [0.2, 0.7], [0.5, 0.85])
    assert rows[0]["far"] == pytest.approx(0.5)
    assert rows[0]["frr"] == pytest.approx(0.0)
    assert rows[1]["far"] == pytest.approx(0.0)
    assert rows[1]["frr"] == pytest.approx(0.5)

