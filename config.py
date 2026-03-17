"""
Central configuration for the AI-Based Traffic Management System.
"""

# Video input (0 for default webcam, or a file path/URL string)
VIDEO_SOURCE = 0  # webcam or file path

# Frame sizing (used by video reader / preprocessing)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Timing configuration (seconds)
BASE_GREEN_TIME = 10
TIME_PER_VEHICLE = 2
YELLOW_TIME = 3
MAX_GREEN_TIME = 60

# Emergency vehicle override (optional)
# Note: The default COCO YOLOv8 models may NOT include an "ambulance" class.
# This feature will work if your model's class names contain one of EMERGENCY_LABELS.
EMERGENCY_MODE = False
# Roboflow dataset labels found in `ambulance.v1i.yolov8/data.yaml`
EMERGENCY_LABELS = ("ambulance", "emergency-vehicle")
EMERGENCY_HOLD_SECONDS = 5

# Hybrid emergency confirmation (recommended for demos)
# Emergency override triggers only when:
# - an emergency vehicle is detected AND
# - (flashing red/blue lights are detected OR operator confirms manually)
EMERGENCY_HYBRID_MODE = True
EMERGENCY_MANUAL_KEY = "e"  # press in the OpenCV window to confirm emergency

# Simple light heuristic thresholds (tune per camera)
EMERGENCY_LIGHT_MIN_PIXELS = 250
EMERGENCY_LIGHT_CHANGE_RATIO = 0.20

