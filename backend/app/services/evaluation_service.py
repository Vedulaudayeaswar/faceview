"""Evaluation metrics calculated from supplied experimental observations."""

from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float
    frr: float


def _safe(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(expected: list[str], predicted: list[str]) -> ClassificationMetrics:
    if len(expected) != len(predicted) or not expected:
        raise ValueError("Expected and predicted lists must have the same non-zero length")
    positive = [value != "UNKNOWN" for value in expected]
    tp = sum(e == p and e != "UNKNOWN" for e, p in zip(expected, predicted))
    fp = sum(e == "UNKNOWN" and p != "UNKNOWN" for e, p in zip(expected, predicted))
    fn = sum(e != "UNKNOWN" and p == "UNKNOWN" for e, p in zip(expected, predicted))
    impostor_attempts = sum(not item for item in positive)
    genuine_attempts = sum(positive)
    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)
    return ClassificationMetrics(
        accuracy=_safe(sum(e == p for e, p in zip(expected, predicted)), len(expected)),
        precision=precision,
        recall=recall,
        f1=_safe(2 * precision * recall, precision + recall),
        far=_safe(fp, impostor_attempts),
        frr=_safe(fn, genuine_attempts),
    )


def detection_rate(detected_faces: int, actual_faces: int) -> float:
    if actual_faces < 0 or detected_faces < 0 or detected_faces > actual_faces:
        raise ValueError("Face counts must satisfy 0 <= detected <= actual")
    return _safe(detected_faces, actual_faces)


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms or any(value < 0 for value in latencies_ms):
        raise ValueError("Provide non-negative latency measurements")
    values = sorted(latencies_ms)
    p95_index = min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)
    midpoint = len(values) // 2
    median = values[midpoint] if len(values) % 2 else (values[midpoint - 1] + values[midpoint]) / 2
    return {"average_ms": sum(values) / len(values), "median_ms": median, "p95_ms": values[p95_index]}


def threshold_analysis(
    genuine_scores: list[float], impostor_scores: list[float], thresholds: list[float]
) -> list[dict[str, float]]:
    """Evaluate threshold tradeoffs from genuine and impostor score samples."""
    if not genuine_scores or not impostor_scores or not thresholds:
        raise ValueError("Genuine scores, impostor scores, and thresholds are required")
    rows = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be between 0 and 1")
        false_accepts = sum(score >= threshold for score in impostor_scores)
        false_rejects = sum(score < threshold for score in genuine_scores)
        true_accepts = len(genuine_scores) - false_rejects
        false_positives = false_accepts
        precision = _safe(true_accepts, true_accepts + false_positives)
        recall = _safe(true_accepts, len(genuine_scores))
        rows.append(
            {
                "threshold": threshold,
                "far": _safe(false_accepts, len(impostor_scores)),
                "frr": _safe(false_rejects, len(genuine_scores)),
                "precision": precision,
                "recall": recall,
                "f1": _safe(2 * precision * recall, precision + recall),
            }
        )
    return rows
