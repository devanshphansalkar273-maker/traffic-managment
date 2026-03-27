"""
Central configuration for the AI-Based Traffic Management System.
"""

# ─── Dual Inference System (YOLOv8 + TensorFlow) ──────────────────────────────
# Set USE_TF to True to use TensorFlow SavedModel inference instead of PyTorch.
# Make sure you've exported the model first:
#   from ultralytics import YOLO
#   model = YOLO("yolov8n.pt")
#   model.export(format="tf")       # for SavedModel
#   model.export(format="tflite")   # for TFLite
USE_TF = False

# Mapping of COCO class indices to class labels for filtering.
# We keep only car, motorcycle, bus, truck
VEHICLE_CLASS_MAPPING = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ─── Demo mode ────────────────────────────────────────────────────────────────
# DEMO_MODE = True  → auto-downloads a real traffic video from YouTube using
#                     yt-dlp and runs real YOLOv8 inference on it.
#                     No webcam needed. First run installs yt-dlp + downloads video.
# DEMO_MODE = False → uses VIDEO_SOURCE below (webcam or your own video file).
DEMO_MODE = True
# ─────────────────────────────────────────────────────────────────────────────

# Video input (used only when DEMO_MODE = False)
# 0 = default webcam, or supply a file path / URL string
VIDEO_SOURCE = 0

# ─── Primary demo video ───────────────────────────────────────────────────────
# YouTube URL for demo video (ambulance footage).
# The primary URL is tried first; if it fails, each fallback is attempted in order.
# Updated to: https://youtu.be/MxUFsufAGoc
DEMO_YOUTUBE_URL = "https://youtu.be/MxUFsufAGoc"   # ambulance footage

# Set DEMO_FORCE_REDOWNLOAD = True to delete the cached video and re-download
# fresh from DEMO_YOUTUBE_URL on the next run. Resets to False automatically
# after the download succeeds (edit manually to force again).
DEMO_FORCE_REDOWNLOAD = True

# Local path where the demo video is saved.
# Delete this file (or set DEMO_FORCE_REDOWNLOAD = True) to trigger a fresh download.
DEMO_VIDEO_PATH = "demo_ambulance.mp4"

# Fallback URLs tried in order if the primary download fails.
DEMO_YOUTUBE_FALLBACKS = [
    "https://www.youtube.com/watch?v=VEz5IXKAiLw",   # ambulance responding with lights
    "https://www.youtube.com/watch?v=W3pSEMvNgk0",   # ambulance siren and driving
    "https://www.youtube.com/watch?v=nt3D26lrkho",   # NYC busy intersection
    "https://www.youtube.com/watch?v=MNn9qKG2UFI",   # original traffic URL
]

# Direct MP4 URLs used as last resort when ffmpeg is not installed.
# The downloader sends browser-like headers so CDN links work too.
DEMO_DIRECT_MP4_FALLBACKS = [
    # Pixabay — no referrer restrictions on direct asset links
    "https://cdn.pixabay.com/video/2020/07/30/46026-447087782_large.mp4",   # city traffic
    "https://cdn.pixabay.com/video/2016/12/30/7026-197634410_large.mp4",    # highway traffic
    # Pexels — works with browser User-Agent headers
    "https://videos.pexels.com/video-files/1197801/1197801-uhd_2560_1440_25fps.mp4",
    "https://videos.pexels.com/video-files/855564/855564-hd_1920_1080_25fps.mp4",
]

# Frame sizing
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# Timing configuration (seconds)
BASE_GREEN_TIME  = 10
TIME_PER_VEHICLE = 2
YELLOW_TIME      = 3
MAX_GREEN_TIME   = 60

# ─── Emergency vehicle ────────────────────────────────────────────────────────
# EMERGENCY_MODE = True enables emergency vehicle detection + signal override.
# Note: default COCO yolov8n.pt does NOT include "ambulance" as a class label.
# In DEMO mode the overlay below simulates the emergency vehicle visually;
# press 'e' in the window to trigger the same override logic manually.
EMERGENCY_MODE         = True
EMERGENCY_LABELS       = ("ambulance", "emergency-vehicle")
EMERGENCY_HOLD_SECONDS = 300

# Hybrid emergency confirmation (light-flash heuristic + manual key)
EMERGENCY_HYBRID_MODE   = True
EMERGENCY_MANUAL_KEY    = "e"   # press in the OpenCV window to confirm

# ── Strobe / light detection thresholds ──────────────────────────────────────
# Detection uses variance (Coefficient of Variation) so headlights are rejected:
#   Steady headlight : CV ~0.02-0.08  (score barely moves frame to frame)
#   Ambulance strobe : CV ~0.40-1.50  (score swings with the flash)
#   Ambulance body   : CV ~0.10-0.25  (some variation from vehicle motion)
#
# EMERGENCY_LIGHT_MIN_PIXELS : min mean brightness (normalised score).
#   Ambulance markings/lights typically score 2-6. Raise to reduce false positives.
# EMERGENCY_STROBE_CV : min CV to confirm flashing.
#   0.35 catches most strobes while rejecting headlights.
#   Lower (0.20) if ambulance lights are still missed.
#   Raise (0.50) if headlights trigger false alarms.
# min mean brightness (normalised score per 100 px area).
# Wider HSV ranges now catch more pixels so scores will be higher.
# Raise if headlights false-trigger. Lower if lights still missed.
EMERGENCY_LIGHT_MIN_PIXELS   = 0.5
EMERGENCY_LIGHT_CHANGE_RATIO = 0.10   # legacy, not used in CV mode

# CV threshold for strobe oscillation.
# Red/blue police-style lights flash fast → high CV.
# Lowered to 0.15 to catch compressed video where flashes are smoothed out.
# Raise to 0.45 if steady coloured objects (traffic lights, signs) false-trigger.
EMERGENCY_STROBE_CV          = 0.15
EMERGENCY_FLASH_PULSES       = 1      # legacy, kept for API compat

# Set True to print live strobe scores to terminal — turn off after tuning.
STROBE_DEBUG = False

# ─── Ambulance classifier (two-stage logic gate) ──────────────────────────────
# Path to trained YOLOv8 weights for ambulance detection.
# Run  python train_ambulance.py  to generate this file from the Roboflow dataset.
AMBULANCE_MODEL_PATH = "ambulance_model.pt"
AMBULANCE_CONFIDENCE = 0.35   # min detection confidence (0.0 – 1.0)

# ─── Emergency vehicle demo overlay (Picture-in-Picture) ─────────────────────
# When DEMO_EMERGENCY_OVERLAY = True, a short ambulance / fire-engine clip is
# downloaded automatically and composited as a PiP in the bottom-right corner
# of the main video at regular intervals.
#
# Disabled here because the main video is already ambulance footage.
# Set to True if you switch back to a traffic video as the main source.
DEMO_EMERGENCY_OVERLAY          = False
DEMO_EMERGENCY_VIDEO_PATH       = "demo_emergency.mp4"   # cached locally
DEMO_EMERGENCY_INTERVAL_SECONDS = 30    # PiP fires every N seconds of playback
DEMO_EMERGENCY_DURATION_SECONDS = 8     # how many seconds the PiP stays visible
DEMO_EMERGENCY_PIP_SCALE        = 0.32  # PiP width as fraction of main frame width

# YouTube URLs for emergency vehicle footage (used only if DEMO_EMERGENCY_OVERLAY = True).
# Tried in order; first successful download wins.
DEMO_EMERGENCY_YOUTUBE_URLS = [
    "https://youtu.be/MxUFsufAGoc",                  # ambulance footage (user-provided)
    "https://www.youtube.com/watch?v=VEz5IXKAiLw",   # ambulance responding with lights
    "https://www.youtube.com/watch?v=W3pSEMvNgk0",   # ambulance siren and driving
    "https://www.youtube.com/watch?v=J8M-DwEMz3Q",   # emergency vehicle POV
    "https://www.youtube.com/watch?v=Qm-mHg3XQLY",   # fire truck + ambulance
]

# Wikimedia Commons fallbacks for emergency clip (no auth, no CDN blocks).
DEMO_EMERGENCY_WIKIMEDIA_URLS = [
    # Real ambulance on a Russian street (7.7s, 720p, WebM)
    "https://upload.wikimedia.org/wikipedia/commons/3/38/Suvorov_Street_%28Korolyov%29.webm",
    # French VSAV emergency vehicle responding (13s, 720p)
    "https://upload.wikimedia.org/wikipedia/commons/4/46/VSAV_des_pompiers_en_urgence.ogg",
    # French VSAV + SMUR ambulance intervention (9.9s, 1080p)
    "https://upload.wikimedia.org/wikipedia/commons/b/b2/VSAV_et_SMUR_en_intervention%2C_Poissy_-_Yvelines.ogg",
    # St John ambulance NZ (8.9s)
    "https://upload.wikimedia.org/wikipedia/commons/8/8b/St_John_ambulance%2C_Dunedin%2C_New_Zealand.ogv",
]