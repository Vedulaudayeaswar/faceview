from threading import Event

from app.services.multi_camera_service import MultiCameraManager


def test_camera_workers_run_independently() -> None:
    manager = MultiCameraManager()
    started = {1: Event(), 2: Event()}

    def worker(camera_id):
        def run(stop: Event):
            started[camera_id].set()
            stop.wait(1)
        return run

    manager.add(1, worker(1))
    manager.add(2, worker(2))
    manager.start(1)
    manager.start(2)
    assert started[1].wait(1) and started[2].wait(1)
    manager.remove(1)
    manager.remove(2)
    assert manager.workers == {}

