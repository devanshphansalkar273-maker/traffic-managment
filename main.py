from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any, Optional
import requests

import config
from ambulance_classifier import AmbulanceClassifier
from detector import (
    count_by_region,
    load_model,
    VehicleDetector,
)
from traffic_logic import TrafficController, calculate_density
from utils import log

api_healthy = True

def send_emergency(lane):
    global api_healthy
    try:
        requests.post("http://127.0.0.1:5000/emergency", json={"lane": lane, "type": "ambulance"}, timeout=0.5)
        if not api_healthy:
            log("Backend connection restored.", level="INFO")
            api_healthy = True
    except Exception:
        if api_healthy:
            log("Failed to send emergency: Backend unreachable.", level="WARNING")
            api_healthy = False

def get_decision():
    global api_healthy
    try:
        response = requests.get("http://127.0.0.1:5000/decision", timeout=0.5)
        if not api_healthy:
            log("Backend connection restored.", level="INFO")
            api_healthy = True
        return response.json()
    except Exception:
        if api_healthy:
            log("API Error: Backend unreachable. Defaulting to WAIT.", level="WARNING")
            api_healthy = False
        return {"action": "WAIT", "lane": None}


# ──────────────────────────────────────────────────────────────────────────────
# Demo video download via yt-dlp
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_ytdlp() -> bool:
    """Install yt-dlp if not already available."""
    try:
        import yt_dlp  # type: ignore
        return True
    except ImportError:
        pass
    log("yt-dlp not found — installing automatically …")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "yt-dlp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log("yt-dlp installed.")
        return True
    except Exception as exc:
        log(f"Could not install yt-dlp: {exc}", level="ERROR")
        return False


def _cleanup(path: str) -> None:
    """Remove a partial / empty download file if it exists."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _download_one_url(url: str, dest: str) -> bool:
    """
    Attempt to download a single YouTube URL to `dest` using yt-dlp.

    Format strategy (no ffmpeg required):
      - Prefer a pre-merged single-file MP4 at <= 720p so no post-processing merge is needed.
      - 'mp4[height<=720]' selects already-muxed progressive MP4s — these never need ffmpeg.
      - Falls back to the best available single-file MP4 if no 720p option exists.

    Returns True on success, False on any failure.
    """
    try:
        import yt_dlp  # type: ignore

        ydl_opts = {
            # IMPORTANT: only select pre-muxed (progressive) files so ffmpeg is never needed.
            # Do NOT request bestvideo+bestaudio — that always requires a merge step.
            "format": "mp4[height<=720]/mp4[height<=480]/mp4/best[ext=mp4]/best",
            "outtmpl": dest,
            "quiet": False,
            "no_warnings": True,
            # Never attempt to merge — if no single-file format found, raise instead.
            "merge_output_format": None,
            "prefer_ffmpeg": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            log(f"Download complete: {dest}")
            return True

        log("Download finished but file looks too small — trying next source.", level="WARNING")
        _cleanup(dest)
        return False

    except Exception as exc:
        log(f"yt-dlp failed for {url}: {exc}", level="WARNING")
        _cleanup(dest)
        return False


def _download_direct_mp4(url: str, dest: str) -> bool:
    """
    Download a direct MP4 URL using urllib (no yt-dlp, no ffmpeg needed).

    Sends browser-like headers so CDNs (Pexels, Pixabay, etc.) don't return 403.
    Shows a simple progress log every 10 MB.
    """
    import urllib.request

    # Mimic a real browser request — many CDNs reject plain urllib User-Agents.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "video/mp4,video/*;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.pexels.com/",
        "Connection": "keep-alive",
    }

    try:
        log(f"Trying direct MP4 download: {url}")
        tmp = dest + ".part"
        req = urllib.request.Request(url, headers=headers)
        chunk = 1024 * 1024        # 1 MB chunks
        progress_every = 10        # log every 10 MB
        downloaded_mb = 0

        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                fh.write(data)
                downloaded_mb += len(data) / (1024 * 1024)
                if int(downloaded_mb) % progress_every == 0 and downloaded_mb >= progress_every:
                    log(f"  … {downloaded_mb:.0f} MB downloaded")

        if os.path.exists(tmp) and os.path.getsize(tmp) > 100_000:
            os.replace(tmp, dest)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            log(f"Direct MP4 download complete: {dest}  ({size_mb:.1f} MB)")
            return True

        _cleanup(tmp)
        log("Direct MP4 file too small — skipping.", level="WARNING")
        return False

    except Exception as exc:
        log(f"Direct MP4 download failed for {url}: {exc}", level="WARNING")
        _cleanup(dest + ".part")
        return False


def _download_youtube_video(url: str, dest: str) -> bool:
    """
    Download a traffic demo video to `dest`.

    Strategy:
      1. Skip entirely if the file already exists and is > 1 MB.
      2. Try the primary YouTube URL first, then DEMO_YOUTUBE_FALLBACKS —
         using a pre-merged format so ffmpeg is never required.
      3. If all YouTube attempts fail (e.g. ffmpeg missing, videos unavailable),
         fall back to DEMO_DIRECT_MP4_FALLBACKS — plain urllib fetch, zero deps.
      4. Return True as soon as one download succeeds; False if everything fails.
    """
    # ── Force re-download if requested in config ──────────────────────────────
    force = getattr(config, "DEMO_FORCE_REDOWNLOAD", False)
    if force and os.path.exists(dest):
        try:
            os.remove(dest)
            log(f"DEMO_FORCE_REDOWNLOAD=True — deleted old video: {dest}")
        except OSError as exc:
            log(f"Could not delete old video ({exc}); proceeding anyway.", level="WARNING")

    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        log(f"Demo video already exists: {dest}  (skipping download)")
        return True

    log("This happens only once. Please wait …")

    # ── Phase 1: YouTube via yt-dlp (no ffmpeg format) ────────────────────────
    if _ensure_ytdlp():
        yt_urls = [url] + list(getattr(config, "DEMO_YOUTUBE_FALLBACKS", []))
        for idx, candidate in enumerate(yt_urls, start=1):
            log(f"YouTube source [{idx}/{len(yt_urls)}]: {candidate}")
            if _download_one_url(candidate, dest):
                return True
            log(f"YouTube source {idx}/{len(yt_urls)} failed — trying next …", level="WARNING")
    else:
        log("yt-dlp unavailable; skipping YouTube sources.", level="WARNING")

    # ── Phase 2: Direct MP4 URLs (urllib only, no yt-dlp / ffmpeg) ───────────
    direct_urls = list(getattr(config, "DEMO_DIRECT_MP4_FALLBACKS", []))
    if direct_urls:
        log("Trying direct MP4 fallback sources (no ffmpeg needed) …")
        for idx, mp4_url in enumerate(direct_urls, start=1):
            log(f"Direct MP4 [{idx}/{len(direct_urls)}]: {mp4_url}")
            if _download_direct_mp4(mp4_url, dest):
                return True
            log(f"Direct MP4 {idx}/{len(direct_urls)} failed.", level="WARNING")

    log("All demo video sources failed.", level="ERROR")
    log(
        "Tip: place any traffic video file named  demo_traffic.mp4  in this folder.",
        level="INFO",
    )
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Emergency vehicle demo video
# ──────────────────────────────────────────────────────────────────────────────

def _get_emergency_video() -> bool:
    """
    Obtain a real ambulance video for the PiP overlay.

    Strategy:
      1. Skip if demo_emergency.mp4 already exists (manual drop-in supported).
      2. Try downloading from DEMO_EMERGENCY_YOUTUBE_URLS via yt-dlp
         using a pre-merged format (no ffmpeg needed).
      3. If all YouTube sources fail, fall back to a synthetically generated
         clip drawn with OpenCV so the demo always has something to show.

    Tip: place any ambulance/emergency clip named  demo_emergency.mp4
         in the project folder and this function will skip entirely.
    """
    dest = config.DEMO_EMERGENCY_VIDEO_PATH

    # ── Skip if already present ───────────────────────────────────────────────
    if os.path.exists(dest) and os.path.getsize(dest) > 50_000:
        log(f"Emergency video already exists: {dest}  (skipping download)")
        return True

    # ── Phase 1: YouTube via yt-dlp (user-provided URL + fallbacks) ──────────
    yt_urls = list(getattr(config, "DEMO_EMERGENCY_YOUTUBE_URLS", []))
    if yt_urls and _ensure_ytdlp():
        log(f"Downloading real ambulance footage from YouTube ({len(yt_urls)} source(s)) ...")
        for idx, url in enumerate(yt_urls, start=1):
            log(f"YouTube source [{idx}/{len(yt_urls)}]: {url}")
            if _download_one_url(url, dest):
                log("Real ambulance video downloaded successfully.")
                return True
            log(f"Source {idx}/{len(yt_urls)} failed — trying next ...", level="WARNING")
    else:
        log("No YouTube URLs configured or yt-dlp unavailable.", level="WARNING")

    # ── Phase 2: Wikimedia Commons (real footage, no auth, no CDN block) ─────
    wiki_urls = list(getattr(config, "DEMO_EMERGENCY_WIKIMEDIA_URLS", []))
    if wiki_urls:
        log("Trying Wikimedia Commons ambulance footage ...")
        for idx, url in enumerate(wiki_urls, start=1):
            log(f"Wikimedia source [{idx}/{len(wiki_urls)}]: {url}")
            if _download_direct_mp4(url, dest):
                log("Real ambulance video downloaded successfully.")
                return True
            log(f"Source {idx}/{len(wiki_urls)} failed — trying next ...", level="WARNING")

    # ── Phase 3: synthetic fallback (always works, no network needed) ─────────
    log("All real sources failed. Generating synthetic ambulance clip as fallback ...", level="WARNING")
    return _synthesize_emergency_clip(dest)


def _synthesize_emergency_clip(dest: str) -> bool:
    """Draw a simple animated ambulance using OpenCV — zero network dependency."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        log("OpenCV/numpy unavailable — cannot generate emergency clip.", level="WARNING")
        return False

    W, H, FPS, FRAMES = 320, 240, 25, 150   # 6-second clip

    out = None
    for fourcc_str in ("mp4v", "XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out    = cv2.VideoWriter(dest, fourcc, FPS, (W, H))
        if out.isOpened():
            break
        out.release()
        out = None

    if out is None or not out.isOpened():
        log("No suitable VideoWriter codec found.", level="WARNING")
        return False

    log(f"Generating synthetic emergency clip -> {dest}")

    def _draw(idx: int) -> "np.ndarray":
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)
        t = idx / FRAMES
        veh_x = int(W * 1.3 - t * W * 1.6)
        veh_y, vw, vh = H // 2 - 20, 120, 55
        bx1, by1, bx2, by2 = veh_x, veh_y, veh_x + vw, veh_y + vh

        for i in range(1, 5):
            x0 = bx2 + i * 14
            cv2.line(frame, (x0, by1 + 10), (x0 + 18, by1 + 10), (60, 60, 60), 2)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (220, 220, 220), -1)
        cv2.rectangle(frame, (bx1 + 12, by1 - 18), (bx1 + vw - 8, by1), (200, 200, 200), -1)
        cv2.rectangle(frame, (bx1 + 16, by1 - 15), (bx1 + 45, by1 - 3), (100, 160, 180), -1)
        cv2.rectangle(frame, (bx1 + 50, by1 - 15), (bx2 - 12, by1 - 3), (100, 160, 180), -1)

        for wx in (bx1 + 18, bx2 - 18):
            cv2.circle(frame, (wx, by2 + 8), 12, (40, 40, 40), -1)
            cv2.circle(frame, (wx, by2 + 8),  5, (90, 90, 90), -1)

        mx, my = bx1 + vw // 2, by1 + vh // 2
        cv2.rectangle(frame, (mx - 3, my - 10), (mx + 3, my + 10), (0, 0, 210), -1)
        cv2.rectangle(frame, (mx - 10, my - 3), (mx + 10, my + 3), (0, 0, 210), -1)

        flash = (idx // 4) % 2
        lc = (0, 0, 230) if flash == 0 else (50, 50, 50)
        rc = (180, 40, 0) if flash == 1 else (50, 50, 50)
        cv2.rectangle(frame, (bx1 + 18, by1 - 26), (bx1 + 36, by1 - 18), lc, -1)
        cv2.rectangle(frame, (bx1 + 40, by1 - 26), (bx1 + 58, by1 - 18), rc, -1)

        cv2.putText(frame, "AMBULANCE", (bx1 + 6, by2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 180), 1)
        sc = (0, 0, 200) if flash == 0 else (180, 40, 0)
        cv2.rectangle(frame, (0, H - 24), (W, H), (20, 20, 20), -1)
        cv2.putText(frame, ">>> EMERGENCY VEHICLE <<<", (10, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sc, 1)
        return frame

    for i in range(FRAMES):
        out.write(_draw(i))
    out.release()

    if os.path.exists(dest) and os.path.getsize(dest) > 5_000:
        log(f"Synthetic emergency clip ready: {dest}  ({os.path.getsize(dest)//1024} KB)")
        return True

    _cleanup(dest)
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Video source helper
# ──────────────────────────────────────────────────────────────────────────────

def _open_video_source(source: Any) -> Optional[Any]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def run() -> int:
    try:
        import cv2  # type: ignore
    except ImportError:
        log("OpenCV not available. Install: pip install opencv-python", level="ERROR")
        return 1

    # ── Resolve video source ──────────────────────────────────────────────────
    if config.DEMO_MODE:
        log("=== DEMO MODE: real YOLOv8 on YouTube traffic video ===")
        ok = _download_youtube_video(config.DEMO_YOUTUBE_URL, config.DEMO_VIDEO_PATH)
        if not ok:
            log("Could not get demo video. Falling back to webcam (VIDEO_SOURCE).", level="WARNING")
            video_source = config.VIDEO_SOURCE
        else:
            video_source = config.DEMO_VIDEO_PATH
        loop_video = True   # loop so demo never stops mid-presentation

        # ── Emergency PiP clip ────────────────────────────────────────────────
        pip_enabled = (
            getattr(config, "DEMO_EMERGENCY_OVERLAY", False)
            and config.EMERGENCY_MODE
        )
        if pip_enabled:
            pip_ok = _get_emergency_video()
            pip_enabled = pip_ok
    else:
        video_source = config.VIDEO_SOURCE
        loop_video   = False
        pip_enabled  = False

    # ── Load YOLO (base COCO model for general vehicle detection) ─────────────
    # VehicleDetector handles YOLOv8 + TensorFlow dual-backend automatically.
    # AmbulanceClassifier also needs a raw PyTorch model for its COCO fallback
    # path (calls model.predict() directly), so we load both.
    log("Loading YOLOv8n model (auto-downloads ~6 MB on first run) …")
    try:
        detector = VehicleDetector()
        backend_note = "TensorFlow" if detector._use_tf else "YOLOv8 (PyTorch)"
        log(f"VehicleDetector ready — backend: {backend_note}")
    except Exception as exc:
        log(f"Failed to load VehicleDetector: {exc}", level="ERROR")
        log("Install Ultralytics: pip install ultralytics", level="INFO")
        return 1

    try:
        coco_model = load_model("yolov8n.pt")
    except Exception as exc:
        log(f"Failed to load COCO model: {exc}", level="ERROR")
        log("Install Ultralytics: pip install ultralytics", level="INFO")
        return 1
    log("Model ready.")

    # ── Load ambulance classifier (two-stage logic gate) ──────────────────────
    # Gate 1: Is it an ambulance?  (trained on Roboflow ambulance-qasjo dataset)
    # Gate 2: Are lights flashing? (strobe oscillation heuristic)
    # Output: "EMERGENCY" / "STANDBY" / "Standard Traffic"
    clf = AmbulanceClassifier(
        model_path        = getattr(config, "AMBULANCE_MODEL_PATH", "ambulance_model.pt"),
        confidence        = getattr(config, "AMBULANCE_CONFIDENCE", 0.35),
        strobe_min_pixels = getattr(config, "EMERGENCY_LIGHT_MIN_PIXELS", 2.0),
        strobe_cv         = getattr(config, "EMERGENCY_STROBE_CV", 0.35),
        flash_pulses      = getattr(config, "EMERGENCY_FLASH_PULSES", 1),
        coco_model        = coco_model,
        debug             = getattr(config, "STROBE_DEBUG", False),
    )

    # ── Open video ────────────────────────────────────────────────────────────
    cap = _open_video_source(video_source)
    if cap is None:
        log(f"Could not open video source: {video_source}", level="ERROR")
        return 1

    controller      = TrafficController()
    process_every_n = 2
    resize_w, resize_h = 640, 480

    frame_idx            = 0
    read_fail_streak     = 0
    status_line          = "OK"
    cached_boxes: list   = []
    cached_labels: list  = []
    cached_vehicle_count = 0
    cached_region_counts = {"north": 0, "south": 0, "east": 0, "west": 0}
    cached_clf_results: list = []   # latest AmbulanceClassifier output

    # ── Multiple emergency vehicle queue (lane -> detection timestamp) ──────────
    # When multiple ambulances are detected, we track each lane's detection time
    # and serve the one with the LONGEST waiting time first (priority-based fairness)
    ambulance_queue: dict[str, float] = {}

    emergency_confirmed_until: float    = 0.0
    emergency_light_prev_score: float | None = None
    manual_emergency_until: float       = 0.0

    last_active: tuple | None = None
    phase_ends_at: float | None = None
    last_fps_t     = time.monotonic()
    fps            = 0.0
    frames_for_fps = 0

    color_map = {"GREEN": (0, 220, 0), "YELLOW": (0, 220, 220), "RED": (0, 0, 220)}

    # ── Emergency PiP state ───────────────────────────────────────────────────
    pip_cap: Any            = None
    pip_active              = False
    pip_next_trigger: float = (
        time.monotonic() + getattr(config, "DEMO_EMERGENCY_INTERVAL_SECONDS", 30)
    )
    pip_ends_at: float      = 0.0
    _pip_direction_cycle    = ["south", "east", "north", "west"]
    _pip_dir_idx            = 0

    if pip_enabled:
        pip_cap = _open_video_source(config.DEMO_EMERGENCY_VIDEO_PATH)
        if pip_cap is None:
            log("Could not open emergency video for PiP — overlay disabled.", level="WARNING")
            pip_enabled = False
        else:
            log("Emergency vehicle PiP overlay ready.")

    log("Running. Press 'q' to quit." + (" Press 'e' for emergency override." if config.EMERGENCY_MODE else ""))

    emergency_sent = False
    decision_applied = False

    try:
        while True:
            ok, frame = cap.read()

            # Loop video in demo mode
            if not ok:
                if loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok:
                    read_fail_streak += 1
                    status_line = f"Read failed ({read_fail_streak})"
                    if read_fail_streak >= 10:
                        log("Video source exhausted; exiting.", level="ERROR")
                        break
                    continue
            read_fail_streak = 0
            frame_idx += 1

            frame = cv2.resize(frame, (resize_w, resize_h))

            # ── Inference via VehicleDetector (YOLOv8 + TensorFlow dual-backend) ───
            do_infer = (frame_idx % process_every_n == 0) or (frame_idx == 1)
            det_result = None
            if do_infer:
                try:
                    det_result    = detector.detect(frame)
                    status_line    = "OK"
                except Exception as exc:
                    status_line = f"Inference error: {type(exc).__name__}"

                # VehicleDetector returns DetectionResult with boxes, labels, regions
                if det_result is not None:
                    cached_boxes           = det_result.debug["boxes"]
                    cached_labels          = det_result.debug["labels"]
                    cached_vehicle_count   = det_result.vehicle_count
                    cached_region_counts   = det_result.debug["regions"]

                # ── Two-stage ambulance logic gate ─────────────────────────────
                # Runs independently of the COCO model above.
                # Gate 1: Is it an ambulance?  → "Standard Traffic" if not
                # Gate 2: Are lights flashing? → "EMERGENCY" or "STANDBY"
                if config.EMERGENCY_MODE:
                    try:
                        clf_results        = clf.classify(frame)
                        cached_clf_results = clf_results
                    except Exception as exc:
                        status_line = f"Classifier error: {type(exc).__name__}"
                        clf_results = cached_clf_results

                    for r in clf_results:
                        if r["status"] == "EMERGENCY":
                            region = count_by_region(frame, [r["box"]])
                            lane = max(region, key=region.get)
                            current_time = time.time()
                            # Add lane to ambulance queue if not already present
                            # This tracks detection time for priority-based serving
                            if lane not in ambulance_queue:
                                ambulance_queue[lane] = current_time
                                log(f"EMERGENCY detected -> {lane.upper()} | "
                                    f"label={r['label']} conf={r['conf']:.2f} | "
                                    f"Queue size: {len(ambulance_queue)}")

            boxes         = cached_boxes
            labels        = cached_labels
            vehicle_count = cached_vehicle_count
            region_counts = cached_region_counts
            density       = calculate_density(region_counts)

            # ── Emergency hybrid confirmation (legacy heuristic, still active) ─
            now = time.monotonic()

            def _lights_score(frame_img, box) -> float:
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame_img.shape[1] - 1, x2)
                y2 = min(frame_img.shape[0] - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    return 0.0
                roi  = frame_img[y1:y2, x1:x2]
                hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                red1 = cv2.inRange(hsv, (0,   120, 120), (10,  255, 255))
                red2 = cv2.inRange(hsv, (160, 120, 120), (180, 255, 255))
                blue = cv2.inRange(hsv, (90,  120, 120), (130, 255, 255))
                return float(cv2.countNonZero(cv2.bitwise_or(red1, red2)) + cv2.countNonZero(blue))

            # ── Multiple emergency vehicle priority selection ─────────────────────
            # Calculate waiting times for all queued ambulances
            # Priority lane = the one with the LONGEST waiting time (anti-starvation)
            priority_lane: str | None = None
            if ambulance_queue:
                current_time = time.time()
                waiting_times = {
                    lane: current_time - detect_time
                    for lane, detect_time in ambulance_queue.items()
                }
                priority_lane = max(waiting_times, key=waiting_times.get)
                emergency_confirmed_until = now + float(config.EMERGENCY_HOLD_SECONDS)

            emergency_confirmed = False
            if config.EMERGENCY_MODE and priority_lane:
                if now < manual_emergency_until:
                    emergency_confirmed = True
                if now < emergency_confirmed_until:
                    emergency_confirmed = True

            # ── Update traffic controller ──────────────────────────────────────
            if config.EMERGENCY_MODE and priority_lane and emergency_confirmed:
                if not emergency_sent:
                    log(f"\n--- 1. Priority Ambulance Selected [{priority_lane.upper()}] ---")
                    log(f"    Waiting times: { {k: f'{v:.1f}s' for k, v in waiting_times.items()} }")
                    log("--- 2. Alert Sent to Backend ---")
                    send_emergency(priority_lane)
                    emergency_sent = True

                decision = get_decision()
                if decision.get("action") == "GIVE_GREEN":
                    served_lane = decision.get("lane", priority_lane)
                    if not decision_applied:
                        log(f"--- 5. SYSTEM UPDATES SIGNAL: Forcing GREEN on {served_lane.upper()}! ---\n")
                        decision_applied = True
                    controller.force_green(served_lane)
                    # Remove served lane from queue
                    if served_lane in ambulance_queue:
                        del ambulance_queue[served_lane]
                        log(f"    Lane {served_lane.upper()} served and removed from queue. Remaining: {list(ambulance_queue.keys())}")
                else:
                    controller.update(density)
            else:
                emergency_sent = False
                decision_applied = False
                controller.update(density)

            # ── Draw: ambulance classifier results ────────────────────────────
            clf.draw(frame, cached_clf_results)

            # ── Draw YOLO bounding boxes (skip boxes already drawn by clf) ────
            clf_boxes = {r["box"] for r in cached_clf_results}
            for (x1, y1, x2, y2), label in zip(boxes, labels):
                if (x1, y1, x2, y2) not in clf_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

            # ── Emergency vehicle PiP overlay ──────────────────────────────────
            if pip_enabled and pip_cap is not None:
                pip_interval = float(getattr(config, "DEMO_EMERGENCY_INTERVAL_SECONDS", 30))
                pip_duration = float(getattr(config, "DEMO_EMERGENCY_DURATION_SECONDS", 8))
                pip_scale    = float(getattr(config, "DEMO_EMERGENCY_PIP_SCALE", 0.32))

                if not pip_active and now >= pip_next_trigger:
                    pip_active       = True
                    pip_ends_at      = now + pip_duration
                    pip_next_trigger = now + pip_interval
                    pip_lane = _pip_direction_cycle[_pip_dir_idx % len(_pip_direction_cycle)]
                    _pip_dir_idx += 1
                    # Add PiP emergency to the queue (tracks waiting time for priority)
                    ambulance_queue[pip_lane] = time.time()
                    manual_emergency_until = now + pip_duration
                    log(f"[PiP] Emergency vehicle approaching from {pip_lane.upper()} | Queue: {list(ambulance_queue.keys())}")

                if pip_active and now >= pip_ends_at:
                    pip_active = False
                    pip_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                if pip_active:
                    ok_pip, pip_frame = pip_cap.read()
                    if not ok_pip:
                        pip_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok_pip, pip_frame = pip_cap.read()

                    if ok_pip and pip_frame is not None:
                        fh, fw  = frame.shape[:2]
                        pip_w   = max(80, int(fw * pip_scale))
                        pip_h   = int(pip_frame.shape[0] * pip_w / max(1, pip_frame.shape[1]))
                        pip_h   = max(60, pip_h)
                        pip_resized = cv2.resize(pip_frame, (pip_w, pip_h))
                        margin  = 8
                        x_off   = max(0, min(fw - pip_w - margin, fw - pip_w))    
                        y_off   = max(0, min(fh - pip_h - margin, fh - pip_h))
                        roi     = frame[y_off:y_off + pip_h, x_off:x_off + pip_w]
                        blended = cv2.addWeighted(pip_resized, 0.88, roi, 0.12, 0)
                        frame[y_off:y_off + pip_h, x_off:x_off + pip_w] = blended
                        cv2.rectangle(frame,
                                      (x_off - 2, y_off - 2),
                                      (x_off + pip_w + 2, y_off + pip_h + 2),
                                      (0, 0, 220), 2)
                        lbl_y = y_off - 6 if y_off > 20 else y_off + pip_h + 16
                        # Show priority lane (highest waiting time) on PiP overlay
                        display_lane = priority_lane if priority_lane else pip_lane
                        cv2.putText(frame, f"EMERGENCY ({display_lane.upper()})",
                                    (x_off, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 220), 1)

            # ── Overlay panel ──────────────────────────────────────────────────
            px, py, pw, ph = 10, 10, 465, 145
            overlay = frame.copy()
            cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            mode_tag = "[DEMO - YouTube Traffic]" if config.DEMO_MODE else "[LIVE]"
            cv2.putText(frame,
                f"{mode_tag}  Vehicles:{vehicle_count}  "
                f"N:{region_counts['north']} S:{region_counts['south']} "
                f"E:{region_counts['east']} W:{region_counts['west']}",
                (px + 8, py + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)

            # Active signal + countdown
            active_dir   = controller.current_green
            active_color = controller.signals[active_dir].color
            for d in ("north", "south", "east", "west"):
                if controller.signals[d].color == "GREEN":
                    active_dir, active_color = d, "GREEN"
                    break
            if active_color != "GREEN":
                for d in ("north", "south", "east", "west"):
                    if controller.signals[d].color == "YELLOW":
                        active_dir, active_color = d, "YELLOW"
                        break

            active_key = (active_dir, active_color)
            if last_active != active_key:
                if active_color == "GREEN":
                    phase_ends_at = now + max(0, int(controller.signals[active_dir].green_seconds))
                elif active_color == "YELLOW":
                    phase_ends_at = now + config.YELLOW_TIME
                else:
                    phase_ends_at = None
                last_active = active_key

            remaining  = max(0, int(round(phase_ends_at - now))) if phase_ends_at else 0
            active_bgr = color_map.get(active_color, (255, 255, 255))
            cv2.putText(frame,
                f"Active: {active_color} {active_dir.upper()} | t-{remaining}s",
                (px + 8, py + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, active_bgr, 2)

            y = py + 80
            for d in ("north", "south", "east", "west"):
                st  = controller.signals[d]
                bgr = color_map.get(st.color, (200, 200, 200))
                extra = f"{st.green_seconds:>2}s" if st.color == "GREEN" else ""
                cv2.putText(frame, f"{d.upper():<5} {st.color:<6} {extra}",
                            (px + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)
                y += 18

            # Logic gate status banner (shows priority lane with multi-emergency support)
            if config.EMERGENCY_MODE:
                statuses = [r["status"] for r in cached_clf_results]
                if "EMERGENCY" in statuses:
                    # Show priority lane or first detected lane with queue info
                    active_lane = priority_lane if priority_lane else (list(ambulance_queue.keys())[0] if ambulance_queue else "?")
                    queue_info = f" (queue: {len(ambulance_queue)})" if len(ambulance_queue) > 1 else ""
                    gate_text = f"GATE: EMERGENCY -> {active_lane.upper()}{queue_info}"
                    gate_bgr  = (0, 0, 255)
                elif "STANDBY" in statuses:
                    gate_text = "GATE: STANDBY — Ambulance present, lights not active"
                    gate_bgr  = (0, 140, 255)
                elif any(s == "Standard Traffic" for s in statuses):
                    gate_text = "GATE: Standard Traffic"
                    gate_bgr  = (0, 200, 0)
                elif priority_lane and now < emergency_confirmed_until:
                    queue_info = f" (queue: {len(ambulance_queue)})" if len(ambulance_queue) > 1 else ""
                    gate_text = f"EMERGENCY ACTIVE -> {priority_lane.upper()}{queue_info}"
                    gate_bgr  = (0, 0, 255)
                elif ambulance_queue:
                    gate_text = f"Emergency in queue — press '{config.EMERGENCY_MANUAL_KEY}' to confirm"
                    gate_bgr  = (0, 165, 255)
                else:
                    gate_text = f"Emergency mode ON (press '{config.EMERGENCY_MANUAL_KEY}')"
                    gate_bgr  = (180, 180, 180)
                cv2.putText(frame, gate_text, (px + 8, py + ph + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, gate_bgr, 1)

            # FPS counter
            frames_for_fps += 1
            now = time.monotonic()
            if now - last_fps_t >= 1.0:
                fps            = frames_for_fps / max(1e-6, now - last_fps_t)
                last_fps_t     = now
                frames_for_fps = 0
            cv2.putText(frame,
                f"FPS:{fps:.1f}  Infer:1/{process_every_n}  {status_line}",
                (px + 8, py + ph + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("AI Traffic Management", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if config.EMERGENCY_MODE and key == ord(str(config.EMERGENCY_MANUAL_KEY).lower()):
                manual_emergency_until = time.monotonic() + float(config.EMERGENCY_HOLD_SECONDS)

    finally:
        cap.release()
        if pip_cap is not None:
            pip_cap.release()
        cv2.destroyAllWindows()
        log("Shutdown complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())