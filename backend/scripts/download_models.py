"""Download the official OpenCV Zoo YuNet and SFace models."""

from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FILES = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = MODELS / name
        if target.exists() and target.stat().st_size > 100_000:
            print(f"Already present: {target}")
            continue
        print(f"Downloading {name}...")
        urlretrieve(url, target)
        print(f"Saved {target}")


if __name__ == "__main__":
    main()
