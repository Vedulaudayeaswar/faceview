# Model provenance and compliance

## Detector

The detector is `yolov8n-face-lindevs.onnx` from the
[lindevs/yolov8-face](https://github.com/lindevs/yolov8-face) release. It is
a YOLOv8n model trained specifically for face detection on WIDERFace. The
application runs it through OpenCV DNN and uses its actual face boxes after
confidence filtering and non-maximum suppression. It never interprets a COCO
`person` class as a face.

The download script verifies the model SHA-256 before allowing it to be used.

## Recognition

Recognition uses `InceptionResnetV1(pretrained="vggface2")` from
[facenet-pytorch](https://github.com/timesler/facenet-pytorch). It is a fixed,
pretrained FaceNet-compatible feature extractor. It receives a padded,
square, 160x160 RGB crop made from a YOLO face bounding box and returns a
normalized 512-dimensional embedding.

## Model/data separation

The FaceNet model is never retrained during enrollment or deletion:

```text
enroll photo -> YOLO face box -> FaceNet embedding -> SQLite + FAISS
delete identity -> deactivate SQLite records + remove FAISS vectors
recognize frame -> YOLO face box -> FaceNet embedding -> FAISS -> threshold
```

The runtime only loads FaceNet-labelled 512-D embeddings into the FAISS index.
Embeddings created by a previous model are kept out of searches so different
embedding spaces are never mixed. Re-enroll a reference photo for each active
identity after a model migration.

## Prohibited model families

This implementation contains no InsightFace package, model zoo, FaceAnalysis
API, Buffalo checkpoint, or Buffalo-derived model. The detector and embedder
are explicitly YOLOv8-Face and FaceNet respectively.
