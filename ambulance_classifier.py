"""
ambulance_classifier.py
=======================
Two-stage logic gate for emergency vehicle status.

  Gate 1 — IDENTIFICATION
      Run the custom-trained YOLOv8 model (trained on Roboflow dataset
      'ambulance-qasjo', classes: ['ambulance', 'emergency-vehicle']).
      If the vehicle is NOT in those classes  →  return "Standard Traffic"

  Gate 2 — VERIFICATION
      Is the emergency light bar / strobe currently flashing?
      Scans only the TOP THIRD of the bounding box (where light bars sit).
      Requires high-saturation colour pixels (rejects white headlights).
      Uses an ON→OFF→ON pulse counter (rejects steady lights).

  Output per detection
      "EMERGENCY"        — ambulance/emergency-vehicle + lights flashing
      "STANDBY"          — ambulance/emergency-vehicle + lights NOT flashing
      "Standard Traffic" — not recognised as an emergency vehicle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Classes that count as emergency vehicles (must match data.yaml names exactly)
EMERGENCY_CLASSES = frozenset({"ambulance", "emergency-vehicle"})

# COCO classes used as fallback Gate 1 when trained model is unavailable.
# Ambulances appear as truck or car in COCO; bus covers larger emergency vehicles.
_COCO_CANDIDATE_LABELS = frozenset({"truck", "bus", "car", "motorcycle"})

# HSV ranges for emergency light colours
# High saturation (>= 130) deliberately excludes pale/white headlights
_LIGHT_RANGES = [
    ((0,   130, 130), (10,  255, 255)),   # red (low hue)
    ((160, 130, 130), (180, 255, 255)),   # red (high hue wrap)
    ((90,  130, 130), (130, 255, 255)),   # blue
    ((15,  130, 130), (35,  255, 255)),   # yellow / amber
    ((10,  130, 130), (18,  255, 255)),   # orange
]


# ── Per-box strobe state machine ──────────────────────────────────────────────

@dataclass
class _StrobeState:
    """Tracks ON/OFF oscillation for a single bounding box."""
    history:     list  = field(default_factory=list)
    high_count:  int   = 0
    low_seen:    bool  = False
    pulse_count: int   = 0

    def update(self, score: float, high_thresh: float, low_thresh: float) -> bool:
        """
        Feed one frame score. Returns True when >= 2 full pulses observed.

        A pulse = HIGH → LOW → HIGH transition.
        Steady headlights stay HIGH every frame → never complete a pulse.
        Strobes alternate HIGH/LOW            → complete pulses quickly.
        """
        self.history.append(score)
        if len(self.history) > 30:
            self.history.pop(0)

        if score >= high_thresh:
            if self.low_seen:
                self.pulse_count += 1
                self.low_seen = False
            self.high_count += 1
        elif score < low_thresh and self.high_count > 0:
            self.low_seen   = True
            self.high_count = 0

        return self.pulse_count >= 2

    def reset(self) -> None:
        self.pulse_count = 0
        self.high_count  = 0
        self.low_seen    = False


# ── Main classifier ───────────────────────────────────────────────────────────

class AmbulanceClassifier:
    """
    Stateful two-stage logic gate.

    Parameters
    ----------
    model_path : str
        Path to the trained YOLOv8 weights (.pt file).
        Trained on Roboflow dataset 'ambulance-qasjo'
        (classes: ambulance, emergency-vehicle).
    confidence : float
        Minimum YOLO confidence to accept a detection (default 0.35).
    strobe_min_pixels : int
        Minimum vivid-colour pixels in the top-third ROI to count as "ON".
    flash_pulses : int
        Full ON→OFF→ON cycles required before declaring lights active.
    """

    def __init__(
        self,
        model_path: str = "ambulance_model.pt",
        *,
        confidence:        float = 0.35,
        strobe_min_pixels: int   = 100,
        flash_pulses:      int   = 2,
        coco_model: Any    = None,
    ) -> None:
        self._conf         = float(confidence)
        self._strobe_high  = float(strobe_min_pixels)
        self._strobe_low   = self._strobe_high * 0.3
        self._flash_pulses = int(flash_pulses)
        self._model        = self._load_model(model_path)
        self._coco_model   = coco_model   # fallback when trained model not available
        # Strobe state keyed by box-centre snapped to 16-px grid (stable across frames)
        self._strobe_states: dict[tuple[int, int], _StrobeState] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, frame: Any) -> list[dict]:
        """
        Run the full two-stage gate on one frame.

        Returns a list of result dicts, one per detection:
        {
            "box":    (x1, y1, x2, y2),
            "label":  "ambulance" | "emergency-vehicle",
            "conf":   float,
            "status": "EMERGENCY" | "STANDBY" | "Standard Traffic",
        }
        """
        try:
            import cv2
        except ImportError:
            return []

        if frame is None:
            return []

        # ── Gate 1: choose detection path ─────────────────────────────────────
        if self._model is not None:
            # Trained ambulance model available — use it directly
            results    = self._model.predict(frame, conf=self._conf, verbose=False)
            detections = self._parse(results)
        elif self._coco_model is not None:
            # COCO fallback — truck/bus/car are ambulance candidates
            results    = self._coco_model.predict(frame, conf=self._conf, verbose=False)
            detections = self._parse(results)
            detections = [
                (box, lbl, cf) for box, lbl, cf in detections
                if lbl in _COCO_CANDIDATE_LABELS
            ]
        else:
            return []

        output = []
        for box, label, conf in detections:
            if self._model is not None:
                # Trained model path: Gate 1 = class name check
                if label not in EMERGENCY_CLASSES:
                    output.append({"box": box, "label": label, "conf": conf,
                                   "status": "Standard Traffic"})
                    continue
                lights_on = self._gate2_strobe(frame, box, cv2)
                status    = "EMERGENCY" if lights_on else "STANDBY"
            else:
                # COCO fallback path: Gate 1 = strobe check
                # Only vehicles with ACTIVE flashing lights pass Gate 1
                lights_on = self._gate2_strobe(frame, box, cv2)
                if not lights_on:
                    output.append({"box": box, "label": label, "conf": conf,
                                   "status": "Standard Traffic"})
                    continue
                label  = "emergency-vehicle"
                status = "EMERGENCY"

            output.append({"box": box, "label": label, "conf": conf, "status": status})

        self._prune_states(detections)
        return output

    def draw(self, frame: Any, results: list[dict]) -> Any:
        """
        Draw bounding boxes + status labels onto the frame (in-place).

        Colour coding:
            EMERGENCY        → red   box + bright red text
            STANDBY          → orange box
            Standard Traffic → green  box
        """
        try:
            import cv2
        except ImportError:
            return frame

        colour_map = {
            "EMERGENCY":        (0,   0,   255),
            "STANDBY":          (0,   140, 255),
            "Standard Traffic": (0,   200, 0),
        }

        for r in results:
            x1, y1, x2, y2 = r["box"]
            status  = r["status"]
            label   = r["label"]
            conf    = r["conf"]
            colour  = colour_map.get(status, (200, 200, 200))
            thickness = 3 if status == "EMERGENCY" else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

            text = f"{label} [{conf:.2f}] — {status}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Flashing red border when EMERGENCY
            if status == "EMERGENCY":
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 2)

        return frame

    # ── Gate 2 internals ──────────────────────────────────────────────────────

    def _gate2_strobe(self, frame: Any, box: tuple, cv2: Any) -> bool:
        """
        Scan the TOP THIRD of the bounding box for flashing coloured lights.
        Returns True when strobe oscillation is confirmed.
        """
        x1, y1, x2, y2 = box
        # Only examine the roof/light-bar zone (top third of box)
        roof_y2 = y1 + max(1, (y2 - y1) // 3)
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(frame.shape[1] - 1, x2)
        y2c = min(frame.shape[0] - 1, roof_y2)

        if x2c <= x1c or y2c <= y1c:
            return False

        roi = frame[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            return False

        import numpy as np
        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        score = 0.0
        for (lo, hi) in _LIGHT_RANGES:
            mask   = cv2.inRange(hsv, lo, hi)
            score += float(cv2.countNonZero(mask))

        # Stable key: box centre snapped to 16-px grid
        key = (((x1 + x2) // 2) & ~15, ((y1 + y2) // 2) & ~15)
        if key not in self._strobe_states:
            self._strobe_states[key] = _StrobeState()

        confirmed = self._strobe_states[key].update(
            score, self._strobe_high, self._strobe_low
        )
        if confirmed:
            self._strobe_states[key].reset()
        return confirmed

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_model(path: str) -> Any:
        try:
            from ultralytics import YOLO  # type: ignore
            model = YOLO(path)
            from utils import log
            log(f"Loaded ambulance model: {path}")
            return model
        except FileNotFoundError:
            from utils import log
            log(
                f"Ambulance model not found at '{path}'. "
                "Run train_ambulance.py first to train on the Roboflow dataset.",
                level="WARNING",
            )
            return None
        except Exception as exc:
            from utils import log
            log(f"Could not load ambulance model: {exc}", level="ERROR")
            return None

    @staticmethod
    def _parse(results: Any) -> list[tuple[tuple, str, float]]:
        """Extract (box, label, conf) from raw YOLO results."""
        if results is None:
            return []
        r0    = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(r0, "boxes", None)
        names = getattr(r0, "names", {})
        if boxes is None:
            return []

        try:
            xyxy  = boxes.xyxy.detach().cpu().tolist()
            cls   = boxes.cls.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist()
        except Exception:
            return []

        out = []
        for box, c, cf in zip(xyxy, cls, confs):
            label = str(names.get(int(c), int(c))).lower()
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            out.append(((x1, y1, x2, y2), label, float(cf)))
        return out

    def _prune_states(self, detections: list) -> None:
        """Remove strobe states for boxes that are no longer detected."""
        active_keys = {
            (((x1 + x2) // 2) & ~15, ((y1 + y2) // 2) & ~15)
            for (x1, y1, x2, y2), _, _ in detections
        }
        gone = [k for k in self._strobe_states if k not in active_keys]
        for k in gone:
            del self._strobe_states[k]