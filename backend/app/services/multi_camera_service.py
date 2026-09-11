"""Independent worker registry for multiple camera streams."""

from dataclasses import dataclass
from threading import Event, Thread
from typing import Callable


@dataclass
class CameraWorker:
    camera_id: int
    target: Callable[[Event], None]
    stop_event: Event
    thread: Thread | None = None


class MultiCameraManager:
    def __init__(self) -> None:
        self.workers: dict[int, CameraWorker] = {}

    def add(self, camera_id: int, target: Callable[[Event], None]) -> None:
        if camera_id in self.workers:
            raise ValueError(f"Camera worker already exists: {camera_id}")
        self.workers[camera_id] = CameraWorker(camera_id, target, Event())

    def start(self, camera_id: int) -> None:
        worker = self.workers[camera_id]
        if worker.thread and worker.thread.is_alive():
            return
        worker.stop_event.clear()
        worker.thread = Thread(target=worker.target, args=(worker.stop_event,), daemon=True, name=f"camera-{camera_id}")
        worker.thread.start()

    def stop(self, camera_id: int, timeout: float = 2.0) -> None:
        worker = self.workers[camera_id]
        worker.stop_event.set()
        if worker.thread:
            worker.thread.join(timeout=timeout)

    def remove(self, camera_id: int) -> None:
        if camera_id in self.workers:
            self.stop(camera_id)
            del self.workers[camera_id]

