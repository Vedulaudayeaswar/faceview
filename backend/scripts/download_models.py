"""Download the face-specific YOLO checkpoint and pretrained FaceNet weights."""

import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT))
FILES = {
    "yolov8n-face-lindevs.onnx": {
        "url": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx",
        "sha256": "8d0bfb0c3383c5bd7a78dd24ef79a21e2aa456619b6ab5e53867092d1c7dc414",
    },
}


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    for name, source in FILES.items():
        target = MODELS / name
        digest = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else ""
        if digest == source["sha256"]:
            print(f"Already present: {target}")
            continue
        print(f"Downloading {name}...")
        urlretrieve(source["url"], target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != source["sha256"]:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum verification failed for {name}")
        print(f"Saved {target}")
    print("Downloading/loading pretrained FaceNet VGGFace2 weights...")
    from app.models.embedder import FaceNetEmbedder

    print(f"Ready: {FaceNetEmbedder().model_name}")


if __name__ == "__main__":
    main()
