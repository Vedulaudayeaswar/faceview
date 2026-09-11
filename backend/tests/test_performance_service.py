from app.services.performance_service import FrameSkipPolicy, PerformanceTracker


def test_frame_skip_and_tracker() -> None:
    policy = FrameSkipPolicy(1)
    assert [policy.should_process() for _ in range(4)] == [True, False, True, False]
    tracker = PerformanceTracker()
    tracker.record(10)
    tracker.record(20)
    summary = tracker.summary()
    assert summary["processed_frames"] == 2
    assert summary["median_ms"] == 15
    assert summary["fps"] > 0

