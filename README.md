# AI-Based Traffic Management System (YOLOv8)

## Project Overview

An **AI-assisted traffic signal controller** that uses computer vision to manage traffic signals intelligently in real time.

- Uses **YOLOv8** to detect vehicles from a webcam or video file.
- Counts vehicles and estimates **traffic density per direction** (North / South / East / West).
- Automatically selects the lane with the highest density while enforcing:
  - **Only one GREEN signal at a time**
  - **Smooth transitions**: GREEN → YELLOW → RED → next GREEN
  - **Fairness**: forces GREEN if a lane has been waiting too long (anti-starvation)
- Optional **emergency vehicle override** support.

---

## Features

- **Real-time vehicle detection** using YOLOv8 with bounding box overlay.
- **Region-based counting**: coarse partition into `north / south / east / west` using bounding box center points.
- **Adaptive green time**: calculated as `BASE_GREEN_TIME + vehicles × TIME_PER_VEHICLE` (capped at a maximum).
- **Smooth signal transitions** with a timed **YELLOW** phase between GREEN and RED.
- **Fairness / starvation prevention**: any lane waiting longer than 60 seconds is automatically served next.
- **Performance mode**: processes every 2nd (or 3rd) frame for smoother FPS on lower-end hardware.

---

## How to Run

### Install dependencies

```bash
pip install opencv-python ultralytics
```

### Configure video source

Edit `config.py`:

```python
VIDEO_SOURCE = 0          # webcam
VIDEO_SOURCE = "video.mp4"  # or a video file path
```

### Start the system

```bash
python main.py
```

Press **`q`** to quit.

---

## Emergency Vehicle Detection (Optional)

In `config.py`:

- Set `EMERGENCY_MODE = True`
- Ensure `EMERGENCY_LABELS` matches your model's class names.

> **Note:** The default COCO `yolov8n.pt` does **not** include an `"ambulance"` class. Emergency override works best with a custom-trained model.

### Hybrid Confirmation (Recommended)

To avoid false positives, the system uses a hybrid rule — emergency override activates only when:

- An emergency vehicle is detected, **AND**
- A **flashing red/blue lights** heuristic is triggered **OR** a **manual operator confirmation** is given.

Manual confirmation: press the key configured by `EMERGENCY_MANUAL_KEY` (default: **`e`**) in the OpenCV window.

---

## Training a Custom Ambulance Model (Roboflow)

Your Roboflow dataset is located at `ambulance.v1i.yolov8/` and includes the classes:

- `ambulance`
- `emergency-vehicle`

Fine-tune a lightweight model with:

```bash
yolo detect train data="ambulance.v1i.yolov8/data.yaml" model=yolov8n.pt imgsz=640 epochs=50 batch=16
```

After training, update `main.py` to use the produced weights:

```
runs/detect/train/weights/best.pt
```