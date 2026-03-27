"""
detector.py — Vehicle Detection Module
AI-Based Traffic Management System

HYBRID INFERENCE ARCHITECTURE
==============================
This module supports two interchangeable inference backends, selected by
`USE_TF` in config.py.  Both backends share identical pre/post-processing
and always return the same `DetectionResult` and `detect_vehicles()` output
format, so traffic_logic.py and main.py are completely unaffected by
whichever backend is active.

┌─────────────────────────────────────────────────────────┐
│             detect_vehicles(frame)                       │
│                     │                                    │
│        ┌────────────┴────────────┐                       │
│   USE_TF=False              USE_TF=True                  │
│   (default)                                              │
│        │                         │                       │
│  YOLOv8 (PyTorch)        TensorFlow backend              │
│  model.predict()         ┌────────────────┐              │
│        │                 │  SavedModel    │              │
│  Ultralytics Result      │  (preferred)   │              │
│  → filter_vehicles()     └───────┬────────┘              │
│        │                         │  fallback             │
│        │                 ┌────────────────┐              │
│        │                 │  TFLite        │              │
│        │                 │  Interpreter   │              │
│        │                 └───────┬────────┘              │
│        └─────────────────────────┘                       │
│                     │                                    │
│              DetectionResult                             │
│     { vehicle_count, boxes, labels, regions }            │
└─────────────────────────────────────────────────────────┘

EXPORTING YOLOV8 TO TENSORFLOW (run once before USE_TF=True)
=============================================================
# Option A — TensorFlow SavedModel (best for server deployment / GPU)
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.export(format="tf")          # creates  yolov8n_saved_model/

# Option B — TFLite (best for edge / mobile / CPU-only targets)
    model.export(format="tflite")      # creates  yolov8n_saved_model.tflite

# Optional — INT8 quantised TFLite (fastest, requires calibration data)
    model.export(format="tflite", int8=True)

WHEN TO USE EACH BACKEND
=========================
USE_TF=False  (YOLOv8 PyTorch) — recommended for development and demo
  ✓ Zero additional export step.
  ✓ Full Ultralytics Result API (NMS, verbose stats, tracking, etc.).
  ✓ Best real-time accuracy/speed trade-off on CPU with yolov8n.

USE_TF=True  (TensorFlow) — for production deployment when:
  ✓ TF Serving / TFLite runtime is already part of your infrastructure.
  ✓ You want a single graph that co-locates with other TF models.
  ✓ Targeting edge hardware with TFLite delegates (XNNPACK, GPU, EdgeTPU).
  ✓ Need to strip PyTorch from the production image.

PERFORMANCE TIPS
================
  • Always use yolov8n (nano) — fastest, lowest memory.
  • Resize input to 640×480 in main.py before calling detect().
  • Use TFLite INT8 on Raspberry Pi / Coral TPU for real-time speed.
  • Skip-frame logic in main.py works transparently with both backends.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ─── Paths / constants ───────────────────────────────────────────────────────
# Produced by model.export(format="tf")     → SavedModel directory
_TF_MODEL_DIR: str = "yolov8n_saved_model"

# Produced by model.export(format="tflite") → single .tflite file
_TFLITE_MODEL_PATH: str = "yolov8n_saved_model.tflite"

# YOLOv8 standard input resolution (must match the export-time imgsz)
_TF_INPUT_SIZE: tuple[int, int] = (640, 640)   # (width, height)

# Default minimum confidence for TF-backend NMS (PyTorch backend uses its own)
_TF_DEFAULT_CONFIDENCE: float = 0.30

# Default IoU threshold for soft-NMS applied to TF backend raw output
_TF_NMS_IOU: float = 0.45

# COCO vehicle classes — must match VEHICLE_CLASS_MAPPING in config.py
_COCO_VEHICLE_IDS: dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
_VEHICLE_LABELS: frozenset[str] = frozenset(_COCO_VEHICLE_IDS.values())


# ─────────────────────────────────────────────────────────────────────────────
# Internal result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DetectionResult:
    """
    Standardised detection result returned by VehicleDetector.detect().

    Both the YOLOv8 and TensorFlow backends populate this same structure so
    downstream code (main.py, traffic_logic.py) never needs to branch.

    Fields
    ------
    vehicle_count   Total number of detected vehicles after filtering.
    debug           Dict with:
                      "boxes"   — list of (x1,y1,x2,y2) int tuples
                      "labels"  — list of class-name strings, same length
                      "regions" — {north/south/east/west: count} dict
    backend         Which inference engine was used ("yolov8" | "savedmodel" | "tflite").
    """
    vehicle_count: int
    debug: dict[str, Any] = field(default_factory=dict)
    backend: str = "yolov8"


# ─────────────────────────────────────────────────────────────────────────────
# Public unified wrapper — this is what main.py calls
# ─────────────────────────────────────────────────────────────────────────────

def detect_vehicles(
    frame: np.ndarray,
    *,
    detector: "VehicleDetector | None" = None,
) -> list[dict[str, Any]]:
    """
    Unified vehicle detection entry point.

    Accepts an OpenCV BGR frame and returns a list of detections in the
    standardised dict format used throughout the system:

        [
            {"bbox": [x1, y1, x2, y2], "class": "car",  "conf": 0.87},
            {"bbox": [x1, y1, x2, y2], "class": "truck", "conf": 0.72},
            ...
        ]

    Args:
        frame:    BGR numpy array (H×W×3) from OpenCV.
        detector: Optional pre-constructed VehicleDetector instance (reuse for
                  performance). If None, a module-level singleton is created.

    Returns:
        List of detection dicts; empty list on error or empty frame.

    Note
    ----
    This function is deliberately backend-agnostic.  The active backend is
    determined by `USE_TF` in config.py at the time VehicleDetector is first
    instantiated.
    """
    if frame is None:
        return []

    _det = detector or _get_singleton_detector()
    result: DetectionResult = _det.detect(frame)

    boxes  = result.debug.get("boxes",  [])
    labels = result.debug.get("labels", [])
    scores = result.debug.get("scores", [])

    # Build standardised list; confidence may be unavailable for some paths
    output: list[dict[str, Any]] = []
    for i, (box, lbl) in enumerate(zip(boxes, labels)):
        conf = float(scores[i]) if i < len(scores) else 0.0
        output.append({
            "bbox":  list(box),   # [x1, y1, x2, y2]
            "class": lbl,
            "conf":  conf,
        })
    return output


# Module-level singleton (created on first call to detect_vehicles)
_SINGLETON_DETECTOR: Optional["VehicleDetector"] = None


def _get_singleton_detector() -> "VehicleDetector":
    global _SINGLETON_DETECTOR
    if _SINGLETON_DETECTOR is None:
        _SINGLETON_DETECTOR = VehicleDetector()
    return _SINGLETON_DETECTOR


# ─────────────────────────────────────────────────────────────────────────────
# TensorFlow helpers (only imported when USE_TF = True)
# ─────────────────────────────────────────────────────────────────────────────

def _tf_backend_available() -> str:
    """
    Return which TF export is present on disk.

    Returns: "saved_model" | "tflite" | "none"
    """
    if os.path.isdir(_TF_MODEL_DIR):
        return "saved_model"
    if os.path.isfile(_TFLITE_MODEL_PATH):
        return "tflite"
    return "none"


def _load_tf_saved_model() -> Any:
    """
    Load a YOLOv8 TensorFlow SavedModel (from model.export(format="tf")).

    The SavedModel is loaded in TF2 eager-execution mode.  It exposes a
    "serving_default" signature that accepts a float32 tensor of shape
    (1, 640, 640, 3) and returns raw detection arrays.
    """
    import tensorflow as tf  # type: ignore
    print(f"[detector] Loading TF SavedModel from '{_TF_MODEL_DIR}' …")
    model = tf.saved_model.load(_TF_MODEL_DIR)
    print("[detector] TF SavedModel loaded successfully.")
    return model


def _load_tf_tflite() -> Any:
    """
    Load a YOLOv8 TFLite model (from model.export(format="tflite")).

    Allocates tensors immediately so the first inference call is fast.
    TFLite delegates (XNNPACK, GPU, EdgeTPU) can be added here if needed.
    """
    import tensorflow as tf  # type: ignore
    print(f"[detector] Loading TFLite model from '{_TFLITE_MODEL_PATH}' …")
    interp = tf.lite.Interpreter(model_path=_TFLITE_MODEL_PATH)
    interp.allocate_tensors()
    print("[detector] TFLite model loaded and tensors allocated.")
    return interp


def _preprocess_for_tf(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a BGR OpenCV frame for TensorFlow YOLOv8 inference.

    Steps:
      1. BGR → RGB  (OpenCV reads BGR; model expects RGB).
      2. Resize to _TF_INPUT_SIZE (640 × 640).
      3. Normalise pixel values to [0.0, 1.0] float32.
      4. Add batch dimension → shape (1, 640, 640, 3).

    Args:
        frame: BGR uint8 array of any resolution.
    Returns:
        Float32 array of shape (1, 640, 640, 3).
    """
    import cv2  # type: ignore
    w, h = _TF_INPUT_SIZE
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h))
    normed  = resized.astype(np.float32) / 255.0
    return np.expand_dims(normed, axis=0)   # (1, H, W, 3)


def _infer_saved_model(
    tf_model: Any,
    input_tensor: np.ndarray,
) -> np.ndarray:
    """
    Run a forward pass through the TF SavedModel.

    The exported YOLOv8 SavedModel uses a "serving_default" signature.
    The primary output is keyed "output0" and has shape (1, 84, 8400):
        84  = 4 bbox coords + 80 COCO class scores
        8400 = total anchor candidates at 640×640 resolution

    Args:
        tf_model:     Loaded tf.saved_model object.
        input_tensor: Float32 array (1, H, W, 3).
    Returns:
        Raw output array (typically shape (1, 84, 8400) or transposed).
    """
    import tensorflow as tf  # type: ignore
    infer  = tf_model.signatures["serving_default"]
    # The export wraps the input under the key used at export time (often "x")
    result = infer(x=tf.constant(input_tensor))
    # Prefer "output0"; fall back to the first output key
    out_key = "output0" if "output0" in result else next(iter(result))
    return result[out_key].numpy()


def _infer_tflite(
    interpreter: Any,
    input_tensor: np.ndarray,
) -> np.ndarray:
    """
    Run a forward pass through the TFLite Interpreter.

    TFLite models may expect uint8 input (INT8-quantised) or float32 (FP16/FP32).
    This function casts automatically based on the input tensor's dtype.

    Args:
        interpreter:  Allocated tf.lite.Interpreter.
        input_tensor: Float32 array (1, H, W, 3).
    Returns:
        Raw output array from the first output tensor.
    """
    in_details  = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    # Auto-cast: INT8-quantised models need uint8 input
    dtype = in_details[0]["dtype"]
    if dtype == np.uint8:
        input_tensor = (input_tensor * 255.0).clip(0, 255).astype(np.uint8)
    elif dtype == np.float16:
        input_tensor = input_tensor.astype(np.float16)
    # else float32 — already correct

    interpreter.set_tensor(in_details[0]["index"], input_tensor)
    interpreter.invoke()
    return interpreter.get_tensor(out_details[0]["index"])


def _nms(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    iou_thresh: float = _TF_NMS_IOU,
) -> list[int]:
    """
    Greedy Non-Maximum Suppression.

    Returns sorted list of surviving indices (highest-confidence first).
    Applied to TF backend raw output because Ultralytics NMS runs inside
    model.predict() but not in the raw TF export.
    """
    if not boxes:
        return []

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []

    while order:
        i = order.pop(0)
        keep.append(i)
        x1, y1, x2, y2 = boxes[i]
        area_i = max(0, x2 - x1) * max(0, y2 - y1)

        surviving: list[int] = []
        for j in order:
            bx1, by1, bx2, by2 = boxes[j]
            inter_x1 = max(x1, bx1)
            inter_y1 = max(y1, by1)
            inter_x2 = min(x2, bx2)
            inter_y2 = min(y2, by2)
            inter_w  = max(0, inter_x2 - inter_x1)
            inter_h  = max(0, inter_y2 - inter_y1)
            inter    = inter_w * inter_h
            area_j   = max(0, bx2 - bx1) * max(0, by2 - by1)
            union    = area_i + area_j - inter
            iou      = inter / union if union > 0 else 0.0
            if iou <= iou_thresh:
                surviving.append(j)
        order = surviving

    return keep


def _postprocess_tf_output(
    raw: np.ndarray,
    *,
    min_confidence: float = _TF_DEFAULT_CONFIDENCE,
    orig_frame_shape: tuple[int, ...] | None = None,
) -> tuple[
    list[tuple[int, int, int, int]],
    list[str],
    list[float],
]:
    """
    Convert raw YOLOv8 TensorFlow output to (boxes, labels, scores).

    YOLOv8 raw export shape: (1, 84, 8400)
        • Columns 0-3   → bbox (cx, cy, w, h) in model-input pixels
        • Columns 4-83  → per-class confidence scores (80 COCO classes)

    Processing pipeline:
      1. Squeeze batch → (8400, 84)  [or detect transposed layout].
      2. Split into bbox and class scores.
      3. Keep only vehicle-class candidates above min_confidence.
      4. Convert cx/cy/w/h → x1/y1/x2/y2 (clipped to [0, input_size]).
      5. Apply greedy NMS to remove duplicates.
      6. Optionally scale boxes back to original frame resolution.

    Args:
        raw:              Raw model output array (any shape variant).
        min_confidence:   Minimum per-class confidence score.
        orig_frame_shape: (H, W, …) of the original frame before resize.
                          When provided, boxes are scaled to original coords.

    Returns:
        boxes:  list of (x1, y1, x2, y2) in pixel coords.
        labels: list of class-name strings (same length as boxes).
        scores: list of confidence floats (same length as boxes).
    """
    # ── 1. Normalise shape ─────────────────────────────────────────────────
    arr = np.squeeze(raw)   # remove batch dim → (84, 8400) or (8400, 84)

    if arr.ndim != 2:
        return [], [], []

    # YOLOv8 SavedModel may export transposed relative to PyTorch layout.
    # Canonical layout: rows = candidates (8400), cols = features (84).
    # If shape is (84, 8400) → transpose.
    rows, cols = arr.shape
    if rows < cols:
        arr = arr.T   # now (8400, 84)

    n_candidates, n_features = arr.shape
    if n_features < 5:
        return [], [], []

    # ── 2. Split bbox / class scores ───────────────────────────────────────
    bbox_raw    = arr[:, :4]         # (N, 4) — cx, cy, w, h
    class_scores = arr[:, 4:]        # (N, n_classes)

    # ── 3. Filter candidates by vehicle class + confidence ─────────────────
    # For each candidate take the argmax across all classes.
    cls_ids      = np.argmax(class_scores, axis=1).astype(int)
    confidences  = np.max(class_scores, axis=1).astype(float)

    boxes_pre:  list[tuple[int, int, int, int]] = []
    labels_pre: list[str]  = []
    scores_pre: list[float] = []

    iw, ih = _TF_INPUT_SIZE   # model coordinate space

    for idx in range(n_candidates):
        conf   = confidences[idx]
        cls_id = cls_ids[idx]
        label  = _COCO_VEHICLE_IDS.get(cls_id)

        # Skip non-vehicle classes and low-confidence detections
        if label is None or conf < min_confidence:
            continue

        # ── 4. Convert cx/cy/w/h → x1/y1/x2/y2 ───────────────────────────
        cx, cy, w, h = bbox_raw[idx]
        x1 = int(np.clip(cx - w / 2, 0, iw))
        y1 = int(np.clip(cy - h / 2, 0, ih))
        x2 = int(np.clip(cx + w / 2, 0, iw))
        y2 = int(np.clip(cy + h / 2, 0, ih))

        if x2 <= x1 or y2 <= y1:
            continue   # degenerate box

        boxes_pre.append((x1, y1, x2, y2))
        labels_pre.append(label)
        scores_pre.append(float(conf))

    # ── 5. NMS ─────────────────────────────────────────────────────────────
    keep = _nms(boxes_pre, scores_pre)
    boxes  = [boxes_pre[i]  for i in keep]
    labels = [labels_pre[i] for i in keep]
    scores = [scores_pre[i] for i in keep]

    # ── 6. Scale to original frame resolution ──────────────────────────────
    if orig_frame_shape and len(orig_frame_shape) >= 2:
        orig_h, orig_w = orig_frame_shape[:2]
        sx = orig_w / iw
        sy = orig_h / ih
        boxes = [
            (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
            for (x1, y1, x2, y2) in boxes
        ]

    return boxes, labels, scores


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8 (PyTorch) helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str = "yolov8n.pt") -> Any:
    """
    Load a YOLOv8 model via Ultralytics.

    Args:
        model_path: Path to .pt weights file (default: "yolov8n.pt").
    Returns:
        Loaded ultralytics.YOLO instance.
    Raises:
        ImportError: if Ultralytics is not installed.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed.  Run: pip install ultralytics"
        ) from exc
    print(f"[detector] Loading YOLOv8 model '{model_path}' …")
    return YOLO(model_path)


def detect_objects(model: Any, frame: np.ndarray) -> Any:
    """
    Run Ultralytics YOLOv8 inference on a frame.

    Args:
        model: Loaded YOLO model from load_model().
        frame: BGR numpy array (any resolution).
    Returns:
        Ultralytics Results list.
    """
    if model is None:
        raise ValueError("model is None — call load_model() first.")
    if frame is None:
        raise ValueError("frame is None.")
    return model.predict(frame, verbose=False)


def _extract_boxes_by_labels(
    results: Any,
    wanted_labels: frozenset[str],
) -> tuple[list[tuple[int, int, int, int]], list[str], list[float]]:
    """
    Extract filtered (boxes, labels, scores) from Ultralytics Results.

    Args:
        results:       Output of model.predict().
        wanted_labels: Set of class-name strings to keep.
    Returns:
        (boxes, labels, scores) — all the same length.
    """
    if results is None:
        return [], [], []

    r0 = results[0] if isinstance(results, (list, tuple)) and results else results
    boxes_obj = getattr(r0, "boxes", None)
    if boxes_obj is None:
        return [], [], []

    names = (
        getattr(r0, "names", None)
        or getattr(getattr(r0, "model", None), "names", None)
        or {}
    )
    xyxy = getattr(boxes_obj, "xyxy", None)
    cls  = getattr(boxes_obj, "cls",  None)
    conf = getattr(boxes_obj, "conf", None)

    if xyxy is None or cls is None:
        return [], [], []

    def _to_list(t: Any) -> list:
        try:
            return t.detach().cpu().tolist()
        except Exception:
            return t.tolist() if hasattr(t, "tolist") else list(t)

    xyxy_list = _to_list(xyxy)
    cls_list  = _to_list(cls)
    conf_list = _to_list(conf) if conf is not None else [0.0] * len(cls_list)

    out_boxes:  list[tuple[int, int, int, int]] = []
    out_labels: list[str]   = []
    out_scores: list[float] = []

    for box, cls_id, score in zip(xyxy_list, cls_list, conf_list):
        label = str(names.get(int(cls_id), int(cls_id)))
        if label not in wanted_labels:
            continue
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        out_boxes.append((x1, y1, x2, y2))
        out_labels.append(label)
        out_scores.append(float(score))

    return out_boxes, out_labels, out_scores


def filter_vehicles(
    results: Any,
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """
    Filter Ultralytics results to vehicle classes only.

    Returns (boxes, labels) — scores are not surfaced here for backward
    compatibility; use _extract_boxes_by_labels() directly if you need them.
    """
    if results is None:
        return [], []
    boxes, labels, _scores = _extract_boxes_by_labels(results, _VEHICLE_LABELS)
    return boxes, labels


def filter_emergency(
    results: Any,
    emergency_labels: set[str] | None = None,
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """
    Filter Ultralytics results to emergency vehicle classes (e.g. "ambulance").

    Useful only when a custom-trained model whose class names include
    those labels is active (the default COCO yolov8n.pt does NOT have
    "ambulance" as a class).
    """
    wanted = frozenset(emergency_labels or {"ambulance"})
    if results is None:
        return [], []
    boxes, labels, _scores = _extract_boxes_by_labels(results, wanted)
    return boxes, labels


# ─────────────────────────────────────────────────────────────────────────────
# Region counting & visualisation helpers (backend-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def count_vehicles(
    filtered: tuple[list, list],
) -> tuple[int, list[tuple[int, int, int, int]]]:
    """
    Count vehicles from (boxes, labels) tuple.

    Args:
        filtered: (boxes, labels) as returned by filter_vehicles().
    Returns:
        (count, boxes)
    """
    if not filtered:
        return 0, []
    try:
        boxes, _labels = filtered
    except Exception:
        return 0, []
    boxes = list(boxes) if boxes is not None else []
    return len(boxes), boxes


def count_by_region(
    frame: Any,
    boxes: list[tuple[int, int, int, int]],
) -> dict[str, int]:
    """
    Assign vehicles to N/S/E/W quadrants based on bounding box centres.

    The frame is split into four quadrants by its centre point.
    A detection is assigned to east/west when the horizontal offset from
    centre exceeds the vertical offset, and north/south otherwise.

    Args:
        frame: OpenCV frame (used for its shape only).
        boxes: List of (x1, y1, x2, y2) bounding boxes.
    Returns:
        Dict with keys "north", "south", "east", "west" and int counts.
    """
    counts: dict[str, int] = {"north": 0, "south": 0, "east": 0, "west": 0}
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


def _draw_boxes(
    frame: Any,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """
    Draw bounding boxes and labels on a frame in-place (no-op if cv2 absent).

    Args:
        frame:     BGR frame to annotate.
        boxes:     List of (x1, y1, x2, y2) ints.
        labels:    List of class-name strings (same length as boxes).
        color:     BGR colour for boxes and text (default: green).
        thickness: Line thickness in pixels.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return

    for (x1, y1, x2, y2), label in zip(boxes, labels):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness,
        )


# ─────────────────────────────────────────────────────────────────────────────
# VehicleDetector — main public class
# ─────────────────────────────────────────────────────────────────────────────

class VehicleDetector:
    """
    Vehicle detector with YOLOv8 + TensorFlow dual-backend support.

    The active backend is selected by `USE_TF` in config.py:
      • USE_TF=False (default) → Ultralytics YOLOv8 (PyTorch)
      • USE_TF=True            → TensorFlow SavedModel or TFLite

    Both backends return the same DetectionResult; all upstream code is
    backend-agnostic.

    Usage
    -----
        detector = VehicleDetector()
        result   = detector.detect(frame)          # any BGR numpy frame
        count    = result.vehicle_count
        boxes    = result.debug["boxes"]
        regions  = result.debug["regions"]

    Or use the module-level shortcut (preferred for simple scripts):
        detections = detect_vehicles(frame)
    """

    def __init__(
        self,
        *,
        model_path:     str   = "yolov8n.pt",
        min_confidence: float = 0.30,
    ) -> None:
        self._min_confidence = float(min_confidence)
        self._model:      Optional[Any] = None
        self._tf_session: Optional[Any] = None
        self._tf_variant: str = "none"   # "saved_model" | "tflite" | "none"

        # Read backend preference from config (at runtime, not import time)
        self._use_tf = self._read_use_tf_flag()

        if self._use_tf:
            self._init_tf_backend()
        else:
            self._model = self._init_yolov8_backend(model_path)

    # ── Init helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _read_use_tf_flag() -> bool:
        """Read USE_TF from config.py; default False if unavailable."""
        try:
            from config import USE_TF  # type: ignore
            return bool(USE_TF)
        except Exception:
            return False

    @staticmethod
    def _init_yolov8_backend(model_path: str) -> Any:
        """
        Load YOLOv8 model.

        Falls back gracefully: if a custom ambulance model exists at
        AMBULANCE_MODEL_PATH, it is NOT loaded here (done in main.py for the
        two-stage classifier).  This function always loads the base detection
        model (yolov8n.pt by default).
        """
        return load_model(model_path)

    def _init_tf_backend(self) -> None:
        """
        Detect and load whichever TF export is present on disk.

        Priority: SavedModel > TFLite.

        Raises
        ------
        ImportError
            If USE_TF=True but no exported model files are found, or if
            TensorFlow is not installed.
        """
        variant = _tf_backend_available()

        if variant == "none":
            # Graceful fallback: warn and switch to PyTorch
            import warnings
            warnings.warn(
                "\n"
                "[detector] USE_TF=True but no TF model found on disk.\n"
                "  Falling back to YOLOv8 (PyTorch) backend.\n"
                "\n"
                "  To enable the TF backend, export the model first:\n"
                "      from ultralytics import YOLO\n"
                "      model = YOLO('yolov8n.pt')\n"
                "      model.export(format='tf')      # SavedModel\n"
                "  or: model.export(format='tflite')  # TFLite\n",
                stacklevel=3,
            )
            self._use_tf = False
            self._model  = load_model("yolov8n.pt")
            return

        try:
            if variant == "saved_model":
                self._tf_session = _load_tf_saved_model()
                self._tf_variant = "saved_model"
            else:
                self._tf_session = _load_tf_tflite()
                self._tf_variant = "tflite"
        except Exception as exc:
            import warnings
            warnings.warn(
                f"[detector] TF backend load failed ({exc}). "
                "Falling back to YOLOv8 (PyTorch).",
                stacklevel=3,
            )
            self._use_tf  = False
            self._tf_variant = "none"
            self._model   = load_model("yolov8n.pt")

    # ── Detection methods ─────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run vehicle detection on a single frame.

        Args:
            frame: BGR numpy array (H×W×3) from OpenCV.
        Returns:
            DetectionResult with vehicle_count, debug info, and backend tag.

        The active backend is transparent to the caller — both paths return
        the same DetectionResult structure.
        """
        if frame is None:
            return DetectionResult(vehicle_count=0, backend=self._active_backend_name())

        if self._use_tf:
            return self._detect_tf(frame)
        return self._detect_yolov8(frame)

    def _detect_yolov8(self, frame: np.ndarray) -> DetectionResult:
        """
        Inference via Ultralytics YOLOv8 (PyTorch) backend.

        • Calls model.predict() → rich Ultralytics Results.
        • Filters to vehicle classes via _extract_boxes_by_labels().
        • Counts, partitions to regions, draws boxes in-place.
        """
        results = detect_objects(self._model, frame)
        boxes, labels, scores = _extract_boxes_by_labels(results, _VEHICLE_LABELS)

        region_counts = count_by_region(frame, boxes)
        _draw_boxes(frame, boxes, labels)

        return DetectionResult(
            vehicle_count=len(boxes),
            debug={
                "boxes":   boxes,
                "labels":  labels,
                "scores":  scores,
                "regions": region_counts,
            },
            backend="yolov8",
        )

    def _detect_tf(self, frame: np.ndarray) -> DetectionResult:
        """
        Inference via TensorFlow backend (SavedModel or TFLite).

        Pipeline:
          1. Preprocess: BGR → RGB, resize to 640×640, normalise, batch.
          2. Forward pass via the active TF variant.
          3. Postprocess raw output + NMS → (boxes, labels, scores).
          4. Scale boxes back to original frame resolution.
          5. Count, region-partition, draw in-place.

        Falls back to YOLOv8 silently if an inference error occurs.
        """
        try:
            import tensorflow as tf  # type: ignore  # noqa: F401
        except ImportError:
            # TF not installed at runtime — silently fall back
            return self._detect_yolov8(frame)

        try:
            input_tensor = _preprocess_for_tf(frame)

            if self._tf_variant == "tflite":
                raw = _infer_tflite(self._tf_session, input_tensor)
            else:
                raw = _infer_saved_model(self._tf_session, input_tensor)

            boxes, labels, scores = _postprocess_tf_output(
                raw,
                min_confidence=self._min_confidence,
                orig_frame_shape=frame.shape,
            )

        except Exception as exc:
            # Any inference failure falls back to YOLOv8
            print(f"[detector] TF inference error ({exc}); falling back to YOLOv8.")
            return self._detect_yolov8(frame)

        region_counts = count_by_region(frame, boxes)
        _draw_boxes(frame, boxes, labels)

        return DetectionResult(
            vehicle_count=len(boxes),
            debug={
                "boxes":   boxes,
                "labels":  labels,
                "scores":  scores,
                "regions": region_counts,
            },
            backend=self._tf_variant,
        )

    # ── Misc ─────────────────────────────────────────────────────────────────

    def _active_backend_name(self) -> str:
        if self._use_tf:
            return self._tf_variant or "tf"
        return "yolov8"

    @property
    def backend(self) -> str:
        """Name of the currently active inference backend."""
        return self._active_backend_name()

    def export_to_tf(
        self,
        *,
        fmt: str = "tf",
        imgsz: int = 640,
    ) -> str:
        """
        Convenience method: export the loaded YOLOv8 model to TensorFlow format.

        This is intended to be called once during setup, not during inference.

        Args:
            fmt:   "tf" for SavedModel or "tflite" for TFLite.
            imgsz: Input size used at export (must match _TF_INPUT_SIZE).
        Returns:
            Path to the exported model directory or file.

        Example
        -------
            detector = VehicleDetector()
            detector.export_to_tf(fmt="tf")      # SavedModel
            detector.export_to_tf(fmt="tflite")  # TFLite
        """
        if self._model is None:
            raise RuntimeError(
                "No YOLOv8 model loaded. "
                "Instantiate VehicleDetector with USE_TF=False first to export."
            )
        print(f"[detector] Exporting YOLOv8 to TensorFlow format='{fmt}' …")
        exported: str = self._model.export(format=fmt, imgsz=imgsz)
        print(f"[detector] Export complete → {exported}")
        print(
            "\nTo use the TF backend, set USE_TF=True in config.py and restart."
        )
        return exported