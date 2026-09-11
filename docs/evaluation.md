# Evaluation

Place separate enrollment and test images under `dataset/enrollment`, `dataset/test/known/<identity>`, and `dataset/test/unknown`. Conditions such as `low_light`, `side_pose`, `low_resolution`, `distance`, and `occlusion` should be recorded separately.

Run threshold analysis only with measured genuine and impostor similarity scores. Report the resulting FAR, FRR, precision, recall, F1, recognition accuracy, detection rate, FPS, average latency, median latency, and p95 latency. Do not use an enrollment image as the primary recognition test image, and do not report invented metrics.

