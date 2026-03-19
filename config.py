"""
Central configuration for the AI-Based Traffic Management System.
"""

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

# YouTube URL for demo video (ambulance footage).
# The primary URL is tried first; if it fails, each fallback is attempted in order.
# Change DEMO_YOUTUBE_URL to any YouTube video URL you prefer.
DEMO_YOUTUBE_URL = "https://youtu.be/MxUFsufAGoc"   # ambulance footage (user-provided)

# Fallback URLs tried in order if the primary download fails.
DEMO_YOUTUBE_FALLBACKS = [
    "https://www.youtube.com/watch?v=VEz5IXKAiLw",   # ambulance responding with lights
    "https://www.youtube.com/watch?v=W3pSEMvNgk0",   # ambulance siren and driving
    "https://www.youtube.com/watch?v=nt3D26lrkho",   # NYC busy intersection
    "https://www.youtube.com/watch?v=MNn9qKG2UFI",   # original traffic URL
]

DEMO_VIDEO_PATH  = "demo_ambulance.mp4"   # saved locally, never re-downloaded

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
EMERGENCY_HOLD_SECONDS = 8

# Hybrid emergency confirmation (light-flash heuristic + manual key)
EMERGENCY_HYBRID_MODE   = True
EMERGENCY_MANUAL_KEY    = "e"   # press in the OpenCV window to confirm

# Light heuristic thresholds
EMERGENCY_LIGHT_MIN_PIXELS   = 200
EMERGENCY_LIGHT_CHANGE_RATIO = 0.15

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