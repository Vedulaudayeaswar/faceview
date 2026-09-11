"""Runtime performance measurements; values are populated only from observations."""

from dataclasses import dataclass, field
import time

from app.services.evaluation_service import latency_summary


@dataclass
class PerformanceTracker:
    started_at: float = field(default_factory=time.perf_counter)
    processed_frames: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        self.processed_frames += 1
        self.latencies_ms.append(latency_ms)

    def summary(self) -> dict[str, float | int]:
        elapsed = time.perf_counter() - self.started_at
        result: dict[str, float | int] = {
            "processed_frames": self.processed_frames,
            "fps": self.processed_frames / elapsed if elapsed else 0.0,
        }
        if self.latencies_ms:
            result.update(latency_summary(self.latencies_ms))
        return result


class FrameSkipPolicy:
    def __init__(self, frame_skip: int = 0) -> None:
        if frame_skip < 0:
            raise ValueError("frame_skip must be non-negative")
        self.frame_skip = frame_skip
        self._frame_number = -1

    def should_process(self) -> bool:
        self._frame_number += 1
        return self._frame_number % (self.frame_skip + 1) == 0

