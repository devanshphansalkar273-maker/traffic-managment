# AI-Based Traffic Management System (YOLOv8)

## Project idea
This project demonstrates a simple **AI-assisted traffic signal controller**:

- Uses **YOLOv8** to detect vehicles from a webcam/video.
- Counts vehicles and estimates **traffic density per direction** (North/South/East/West).
- Selects the lane with the highest density while enforcing:
  - **Only one GREEN at a time**
  - **Smooth transitions**: GREEN → YELLOW → RED → next GREEN
  - **Fairness**: forces GREEN if a lane waits too long (anti-starvation)
- Optional **emergency override** (e.g., ambulance) if your model supports that label.

## How to run

### Install dependencies
From this folder:

```bash
pip install opencv-python ultralytics
```

### Configure video source
Edit `config.py`:

- `VIDEO_SOURCE = 0` for webcam
- or set it to a file path, e.g. `VIDEO_SOURCE = "video.mp4"`

### Start the demo

```bash
python main.py
```

- Press **q** to quit.

## Features (what to explain in a hackathon)
- **Real-time vehicle detection** (YOLOv8) + bounding boxes overlay.
- **Region-based counting**: coarse partition into `north/south/east/west` using box center points.
- **Adaptive green time**: \(BASE\_GREEN\_TIME + vehicles \times TIME\_PER\_VEHICLE\) (capped).
- **Smooth signal transitions** with a timed **YELLOW** phase.
- **Fairness / starvation prevention**: if any lane waits \(> 60s\), it gets served.
- **Performance mode**: processes every 2nd (or 3rd) frame for smoother FPS.

## Emergency vehicle detection (optional)
In `config.py`:

- Set `EMERGENCY_MODE = True`
- Ensure `EMERGENCY_LABELS` matches your model’s class names

Note: the default COCO `yolov8n.pt` often does **not** include an `"ambulance"` class. Emergency override works best with a **custom-trained model**.

### Hybrid confirmation (recommended)
To avoid triggering on every detected ambulance (which doesn’t guarantee a real emergency), the demo uses a **hybrid rule**:

- Emergency vehicle detected **AND**
- (**flashing red/blue lights** heuristic **OR** **manual operator confirmation**)

Manual confirmation: press the key configured by `EMERGENCY_MANUAL_KEY` (default `e`) in the OpenCV window.

## Training a custom ambulance model (Roboflow export)
Your Roboflow dataset is located at `ambulance.v1i.yolov8/` and includes classes:

- `ambulance`
- `emergency-vehicle`

Train (fine-tune) a lightweight model:

```bash
yolo detect train data="ambulance.v1i.yolov8/data.yaml" model=yolov8n.pt imgsz=640 epochs=50 batch=16
```

After training, use the produced weights in `main.py`:

- `runs/detect/train/weights/best.pt`

