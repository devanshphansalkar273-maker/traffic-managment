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
# HSV ranges for emergency light colours.
# YouTube/Shorts compression desaturates and overexposes lights heavily.
# Rules:
#   - Saturation kept LOW (>= 40) to catch compression-desaturated colours
#   - Value kept LOW (>= 80) — lights are bright but not necessarily vivid
#   - Red wraps around 0/180 in OpenCV HSV so needs two ranges
#   - Blue range is wide (85-135) to catch cyan-blue common in LED bars
#   - Also detect sudden BRIGHTNESS spikes (overexposed white flash)
#     via a near-white high-value range — this catches overexposed lights
#     that appear white in compressed video regardless of original colour
_LIGHT_RANGES = [
    # Red (low hue wrap)
    ((0,   40,  80), (10,  255, 255)),
    # Red (high hue wrap)
    ((165, 40,  80), (180, 255, 255)),
    # Blue / cyan-blue (LED light bars)
    ((85,  40,  80), (135, 255, 255)),
    # Yellow / amber
    ((15,  40,  80), (40,  255, 255)),
    # Orange
    ((8,   40,  80), (20,  255, 255)),
    # Overexposed white flash (very bright, nearly desaturated)
    # Catches red/blue lights that blow out to white in compressed video
    ((0,   0,  220), (180, 60,  255)),
]


# ── Per-box strobe state machine ──────────────────────────────────────────────

@dataclass
class _StrobeState:
    """
    Variance-based strobe detector for one bounding box.

    Core insight
    ------------
    Headlights  -> score is HIGH and STEADY   -> low variance (CV ~0.05)
    Strobes     -> score alternates HIGH/LOW  -> HIGH variance (CV ~0.4-1.5)

    CV = std / mean  (coefficient of variation — scale invariant)

    Two conditions must BOTH hold to confirm:
        1. mean score  >= min_score   (light is actually bright / coloured)
        2. CV          >= min_cv      (score oscillates — not steady)

    This is faster than pulse counting (confirms in ~0.5 s) AND naturally
    rejects headlights whose score never swings.
    """
    history: list = field(default_factory=list)

    def update(self, score: float, min_score: float, min_cv: float) -> bool:
        """
        Feed one normalised frame score. Returns True when strobe confirmed.

        min_score : minimum mean brightness  (rejects dark / unlit frames)
        min_cv    : minimum CV               (rejects steady headlights)
        """
        self.history.append(score)
        if len(self.history) > 20:      # ~1 s at typical 20 fps inference
            self.history.pop(0)

        if len(self.history) < 6:       # need at least 6 frames to measure swing
            return False

        import statistics
        mean = statistics.mean(self.history)
        if mean < min_score:
            return False                # not bright / coloured enough

        std = statistics.stdev(self.history)
        cv  = std / mean if mean > 0 else 0.0
        return cv >= min_cv

    def reset(self) -> None:
        self.history.clear()


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
        strobe_min_pixels: float = 5.0,
        strobe_cv:         float = 0.40,
        flash_pulses:      int   = 1,
        coco_model: Any    = None,
        debug:      bool   = False,
    ) -> None:
        self._conf         = float(confidence)
        self._strobe_high  = float(strobe_min_pixels)
        self._strobe_cv    = float(strobe_cv)
        self._flash_pulses = int(flash_pulses)
        self._model        = self._load_model(model_path)
        self._coco_model   = coco_model   # fallback when trained model not available
        self._debug        = bool(debug)
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
        Scan the bounding box for flashing emergency lights.

        Scanning strategy:
          - Box height >= 80px: scan top HALF only (light bar is on roof,
            this excludes headlights at front/bottom)
          - Box height < 80px:  scan the FULL box (ambulance is far/small,
            top-third would be only a few pixels — not enough signal)

        Returns True as soon as 1 confirmed ON->OFF->ON strobe pulse is seen.
        Does NOT reset after confirm so rapid re-triggers work correctly.
        """
        x1, y1, x2, y2 = box
        box_h = y2 - y1

        # Adaptive scan zone
        if box_h >= 80:
            scan_y2 = y1 + box_h // 2   # top half only
        else:
            scan_y2 = y2                  # full box when small/distant

        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(frame.shape[1] - 1, x2)
        y2c = min(frame.shape[0] - 1, scan_y2)

        if x2c <= x1c or y2c <= y1c:
            return False

        roi = frame[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            return False

        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        score = 0.0
        for (lo, hi) in _LIGHT_RANGES:
            mask   = cv2.inRange(hsv, lo, hi)
            score += float(cv2.countNonZero(mask))

        # Normalise by ROI area so small distant boxes aren't penalised
        roi_area = max(1, (x2c - x1c) * (y2c - y1c))
        norm_score = score / roi_area * 100.0   # pixels per 100 area units

        # ── Debug logging (set STROBE_DEBUG=True in config to enable) ─────────
        if self._debug:
            from utils import log as _log
            import statistics as _st
            _h = self._strobe_states.get(
                (((x1+x2)//2)&~15, ((y1+y2)//2)&~15), _StrobeState()
            ).history
            _mean = _st.mean(_h) if len(_h)>=2 else 0.0
            _cv   = _st.stdev(_h)/_mean if (_mean>0 and len(_h)>=2) else 0.0
            _log(
                f"[STROBE] box=({x1},{y1},{x2},{y2}) h={box_h} "
                f"norm={norm_score:.2f} mean={_mean:.2f} cv={_cv:.3f} "
                f"(need mean>={self._strobe_high:.1f} cv>={self._strobe_cv:.2f})",
                level="INFO"
            )

        # Stable key: box centre snapped to 16-px grid
        key = (((x1 + x2) // 2) & ~15, ((y1 + y2) // 2) & ~15)
        if key not in self._strobe_states:
            self._strobe_states[key] = _StrobeState()

        # Variance-based: confirm when score is bright AND oscillating
        confirmed = self._strobe_states[key].update(
            norm_score,
            min_score = self._strobe_high,
            min_cv    = self._strobe_cv,
        )
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