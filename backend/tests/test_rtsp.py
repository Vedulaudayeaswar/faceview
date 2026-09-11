import pytest

from app.services.camera_service import validate_rtsp_url


def test_rtsp_url_validation() -> None:
    value = validate_rtsp_url("rtsp://user:secret@example.com/live")
    assert value.startswith("rtsp://")


@pytest.mark.parametrize("value", ["http://example.com/live", "not-a-url", "rtsp:///missing-host"])
def test_invalid_rtsp_url(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid RTSP URL"):
        validate_rtsp_url(value)

