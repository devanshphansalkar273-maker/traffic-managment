from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DetectionResult:
    vehicle_count: int
    debug: Optional[dict[str, Any]] = None


_VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}


def load_model(model_name: str = "yolov8n.pt"):
    """
    Load an Ultralytics YOLOv8 model.

    Args:
        model_name: Pretrained model name/path (default: "yolov8n.pt").
    Returns:
        A loaded `ultralytics.YOLO` model instance.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc

    return YOLO(model_name)


def detect_objects(model: Any, frame: Any):
    """
    Run YOLO inference on a frame.

    Args:
        model: Loaded YOLO model returned by `load_model()`.
        frame: Image/frame (e.g., a numpy array in BGR from OpenCV).
    Returns:
        Ultralytics results object/list (raw predictions) for downstream parsing.
    """
    if model is None:
        raise ValueError("model is None; call load_model() first.")
    if frame is None:
        raise ValueError("frame is None.")

    return model.predict(frame, verbose=False)


def _extract_boxes_by_labels(results: Any, wanted_labels: set[str]):
    """
    Extract boxes+labels from YOLO results for a set of class names.
    """
    if results is None:
        return [], []

    r0 = results[0] if isinstance(results, (list, tuple)) and results else results
    boxes_obj = getattr(r0, "boxes", None)
    if boxes_obj is None:
        return [], []

    names = getattr(r0, "names", None) or getattr(getattr(r0, "model", None), "names", None) or {}
    xyxy = getattr(boxes_obj, "xyxy", None)
    cls = getattr(boxes_obj, "cls", None)
    if xyxy is None or cls is None:
        return [], []

    try:
        xyxy_list = xyxy.detach().cpu().tolist()  # type: ignore[attr-defined]
    except Exception:
        xyxy_list = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)

    try:
        cls_list = cls.detach().cpu().tolist()  # type: ignore[attr-defined]
    except Exception:
        cls_list = cls.tolist() if hasattr(cls, "tolist") else list(cls)

    out_boxes: list[tuple[int, int, int, int]] = []
    out_labels: list[str] = []
    for box, cls_id in zip(xyxy_list, cls_list):
        label = str(names.get(int(cls_id), int(cls_id)))
        if label not in wanted_labels:
            continue
        x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        out_boxes.append((x1, y1, x2, y2))
        out_labels.append(label)

    return out_boxes, out_labels


def filter_vehicles(results: Any):
    """
    Filter YOLO results to only vehicle classes (car, motorcycle, bus, truck).

    Args:
        results: The raw object returned by `detect_objects()` / `model.predict()`.
    Returns:
        (boxes, labels)
        - boxes: list of (x1, y1, x2, y2) ints in pixel coordinates
        - labels: list of string class labels (same length as boxes)
    """
    if results is None:
        return [], []

    return _extract_boxes_by_labels(results, _VEHICLE_LABELS)


def filter_emergency(results: Any, emergency_labels: set[str] | None = None):
    """
    Filter YOLO results to emergency vehicle classes (e.g., "ambulance").

    This will only return detections if your model's class names include those labels.
    """
    wanted = emergency_labels or {"ambulance"}
    return _extract_boxes_by_labels(results, wanted)


def count_vehicles(filtered_results: Any) -> tuple[int, list[tuple[int, int, int, int]]]:
    """
    Count vehicles from filtered results.

    Args:
        filtered_results: Output from `filter_vehicles()` i.e. (boxes, labels).
    Returns:
        (count, boxes) where:
        - count: total number of vehicles in the frame
        - boxes: list of vehicle bounding boxes for visualization
    """
    if not filtered_results:
        return 0, []

    try:
        boxes, _labels = filtered_results
    except Exception:
        return 0, []

    boxes = list(boxes) if boxes is not None else []
    return len(boxes), boxes


def count_by_region(frame: Any, boxes: list[tuple[int, int, int, int]]) -> dict[str, int]:
    """
    Count vehicles by coarse region using bounding box center points.

    Regions are a 4-way partition of the frame into north/south/east/west.
    A detection is assigned to exactly one region:
    - If horizontal offset dominates: west/east
    - Otherwise: north/south
    """
    counts = {"north": 0, "south": 0, "east": 0, "west": 0}
    if frame is None or not boxes:
        return counts

    shape = getattr(frame, "shape", None)
    if not shape or len(shape) < 2:
        return counts

    height, width = int(shape[0]), int(shape[1])
    if width <= 0 or height <= 0:
        return counts

    cx0, cy0 = width / 2.0, height / 2.0
    for (x1, y1, x2, y2) in boxes:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = cx - cx0
        dy = cy - cy0

        if abs(dx) > abs(dy):
            counts["east" if dx >= 0 else "west"] += 1
        else:
            counts["south" if dy >= 0 else "north"] += 1

    return counts


def _draw_boxes(frame: Any, boxes: list[tuple[int, int, int, int]], labels: list[str]) -> None:
    """
    Draw bounding boxes + labels on the frame (no-op if OpenCV unavailable).
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return

    for (x1, y1, x2, y2), label in zip(boxes, labels):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


class VehicleDetector:
    """
    Minimal wrapper around YOLOv8 detection.

    This keeps the rest of the project decoupled from the underlying model API.
    """

    def __init__(self, *, min_confidence: float = 0.3) -> None:
        self._min_confidence = float(min_confidence)
        self._model = load_model("yolov8n.pt")

    def detect(self, frame: Any) -> DetectionResult:
        """
        Args:
            frame: A single video frame (typically a numpy array if using OpenCV).
        Returns:
            DetectionResult with at least a vehicle_count.
        """
        results = detect_objects(self._model, frame)
        boxes, labels = filter_vehicles(results)
        count, _boxes_for_viz = count_vehicles((boxes, labels))
        region_counts = count_by_region(frame, boxes)
        _draw_boxes(frame, boxes, labels)
        return DetectionResult(
            vehicle_count=count,
            debug={"boxes": boxes, "labels": labels, "regions": region_counts},
        )

