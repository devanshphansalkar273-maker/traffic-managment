# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

codex gonna review your work
## Project Overview

AI-based traffic signal controller using YOLOv8 computer vision to manage 4 traffic signals (north/south/east/west). Selects the green signal based on vehicle density with anti-starvation protection.

codex gonna review your work
## Running

```bash
python main.py
```

- **Demo mode** (`DEMO_MODE = True` in config.py): Auto-downloads a traffic video and runs real YOLOv8 inference. No webcam required.
- **Webcam/Video mode** (`DEMO_MODE = False`): Uses `VIDEO_SOURCE` in config.py (0 = webcam, or a file path).
- Press `q` to quit. Press `e` to trigger emergency override (when `EMERGENCY_MODE = True`).

## Dependencies

```bash
pip install opencv-python ultralytics
```

For demo video download (optional, used only in DEMO_MODE):
```bash
pip install yt-dlp
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, video capture, main loop, drawing overlays |
| `config.py` | All configuration constants (timing, thresholds, video sources) |
| `detector.py` | YOLOv8 wrapper: `VehicleDetector` class, filtering, region counting |
| `traffic_logic.py` | `TrafficController` class managing signal state machine |
| `utils.py` | Logging, `clamp`, `FrameInfo` dataclass |
| `ambulance_classifier.py` | Two-stage emergency vehicle classifier (separate YOLO + strobe detection) |

## Architecture

### Signal State Machine (traffic_logic.py)

`TrafficController` enforces exactly one GREEN signal at a time with smooth transitions:
- GREEN → YELLOW (3s, configurable) → RED → next GREEN
- Green time = `BASE_GREEN_TIME + vehicles × TIME_PER_VEHICLE` (capped at `MAX_GREEN_TIME`)
- Anti-starvation: any lane waiting >60s gets forced GREEN regardless of density

### Vehicle Detection Pipeline

1. `VehicleDetector.detect(frame)` runs YOLOv8, filters for vehicle classes (`car`, `motorcycle`, `bus`, `truck`)
2. `count_by_region()` assigns each detection to N/S/E/W based on bounding box center relative to frame center
3. Result: `DetectionResult(vehicle_count, debug={"boxes", "labels", "regions"})`

### Emergency Vehicle Detection

Two-stage classifier in `ambulance_classifier.py`:
- **Stage 1**: YOLO check for ambulance/emergency-vehicle labels (requires custom-trained model, default COCO yolov8n.pt lacks "ambulance")
- **Stage 2**: Strobe light oscillation detection via HSV color segmentation and CV (Coefficient of Variation) scoring

Hybrid confirmation mode requires both a YOLO detection AND flashing lights, or a manual `e` key press.

### Backend API Integration (main.py)

When emergency is confirmed, the system sends alerts and polls for decisions:
- `POST /emergency` — notifies backend of ambulance direction
- `GET /decision` — expects `{"action": "GIVE_GREEN", "lane": "north"}` or `{"action": "WAIT"}`
- Falls back to local `controller.force_green()` if backend is unreachable

## Configuration Notes

- **Performance mode**: `process_every_n = 2` (in main.py) skips frames for lower-end hardware
- **Ambulance model**: Set `AMBULANCE_MODEL_PATH` to custom Roboflow-trained weights. Train with: `yolo detect train data="ambulance.v1i.yolov8/data.yaml" model=yolov8n.pt imgsz=640 epochs=50 batch=16`
- **Strobe tuning**: `EMERGENCY_STROBE_CV` (default 0.25) — raise if headlights false-trigger, lower if ambulance lights are missed
