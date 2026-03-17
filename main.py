from __future__ import annotations

from typing import Any, Optional
import time

import config
from detector import (
    count_by_region,
    count_vehicles,
    detect_objects,
    filter_emergency,
    filter_vehicles,
    load_model,
)
from traffic_logic import TrafficController, calculate_density
from utils import log


def _open_video_source(source: Any) -> Optional[Any]:
    """
    Opens a video source using OpenCV if available.

    Returns None if OpenCV isn't installed or the source cannot be opened.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def run() -> int:
    try:
        import cv2  # type: ignore
    except Exception:
        log("OpenCV not available. Install it with: pip install opencv-python", level="ERROR")
        return 1

    # Load lightweight YOLO model once (keeps the loop fast).
    # If you trained a custom model, point this at your `best.pt`.
    try:
        model = load_model("yolov8n.pt")
    except Exception as exc:
        log(f"Failed to load YOLO model: {exc}", level="ERROR")
        log("Install Ultralytics with: pip install ultralytics", level="INFO")
        return 1
    controller = TrafficController()

    cap = _open_video_source(config.VIDEO_SOURCE)
    if cap is None:
        log(
            "Video source failed to open. Check VIDEO_SOURCE in config.py.",
            level="ERROR",
        )
        return 1

    log("Starting traffic management loop. Press 'q' to quit.")

    # Performance knobs for demo smoothness.
    process_every_n = 2  # set to 3 for even smoother FPS on slower machines
    resize_w, resize_h = 640, 480

    frame_idx = 0
    read_fail_streak = 0
    status_line = "OK"
    cached_boxes: list[tuple[int, int, int, int]] = []
    cached_labels: list[str] = []
    cached_vehicle_count = 0
    cached_region_counts = {"north": 0, "south": 0, "east": 0, "west": 0}
    emergency_active_until: float = 0.0
    emergency_direction: str | None = None
    emergency_confirmed_until: float = 0.0
    emergency_light_prev_score: float | None = None
    manual_emergency_until: float = 0.0

    last_active: tuple[str, str] | None = None  # (direction, color)
    phase_ends_at: float | None = None
    last_fps_t = time.monotonic()
    fps = 0.0
    frames_for_fps = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                read_fail_streak += 1
                status_line = f"Camera read failed ({read_fail_streak})"
                if read_fail_streak >= 10:
                    log("Camera read repeatedly failed; exiting.", level="ERROR")
                    break
                # Try again on next iteration.
                continue
            read_fail_streak = 0

            frame_idx += 1

            # Resize for predictable processing cost.
            frame = cv2.resize(frame, (resize_w, resize_h))

            # Detect + post-process (optionally skip frames to maintain FPS).
            do_infer = (frame_idx % process_every_n == 0) or (frame_idx == 1)
            if do_infer:
                try:
                    results = detect_objects(model, frame)
                except Exception as exc:
                    # Don't crash the demo; keep the last cached detections.
                    status_line = f"Inference error: {type(exc).__name__}"
                    results = None

                # Optional emergency vehicle detection (requires a model with matching labels).
                if config.EMERGENCY_MODE and results is not None:
                    e_boxes, e_labels = filter_emergency(results, set(config.EMERGENCY_LABELS))
                    if e_boxes:
                        e_regions = count_by_region(frame, e_boxes)
                        emergency_direction = max(e_regions, key=e_regions.get)
                        # Candidate detected; confirmation handled below (hybrid mode).
                        emergency_active_until = time.monotonic() + float(config.EMERGENCY_HOLD_SECONDS)

                if results is not None:
                    boxes, labels = filter_vehicles(results)
                    vehicle_count, _boxes_for_viz = count_vehicles((boxes, labels))
                    region_counts = count_by_region(frame, boxes)

                    cached_boxes = boxes
                    cached_labels = labels
                    cached_vehicle_count = vehicle_count
                    cached_region_counts = region_counts
                    status_line = "OK"

            boxes = cached_boxes
            labels = cached_labels
            vehicle_count = cached_vehicle_count
            region_counts = cached_region_counts

            density = calculate_density(region_counts)

            # --- Emergency hybrid confirmation ---
            # We can't know if there's a patient inside; we can only approximate:
            # - visual cue (flashing red/blue lights), OR
            # - manual operator confirmation (hackathon demo-friendly).
            now = time.monotonic()

            def _lights_score(frame_img, box) -> float:
                # Returns a rough "red/blue pixels" count inside the box.
                try:
                    import cv2  # type: ignore
                except Exception:
                    return 0.0
                x1, y1, x2, y2 = box
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_img.shape[1] - 1, x2)
                y2 = min(frame_img.shape[0] - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    return 0.0

                roi = frame_img[y1:y2, x1:x2]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Red wraps hue → combine two masks.
                red1 = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
                red2 = cv2.inRange(hsv, (160, 120, 120), (180, 255, 255))
                red = cv2.bitwise_or(red1, red2)

                blue = cv2.inRange(hsv, (90, 120, 120), (130, 255, 255))
                return float(cv2.countNonZero(red) + cv2.countNonZero(blue))

            emergency_confirmed = False
            if config.EMERGENCY_MODE and emergency_direction:
                # Manual confirmation (press key in the OpenCV window).
                if now < manual_emergency_until:
                    emergency_confirmed = True

                # Visual confirmation (flashing lights heuristic).
                # Best effort: measure red/blue pixels, and look for changes over time.
                if not emergency_confirmed and config.EMERGENCY_HYBRID_MODE:
                    # Use last known emergency boxes if available in this frame:
                    # (we don't store them separately; reuse vehicle boxes if labels match is hard)
                    # So we only compute this during inference frames when results exist.
                    if do_infer and config.EMERGENCY_MODE:
                        try:
                            e_boxes_now, _ = filter_emergency(results, set(config.EMERGENCY_LABELS)) if results is not None else ([], [])
                        except Exception:
                            e_boxes_now = []

                        if e_boxes_now:
                            score = _lights_score(frame, e_boxes_now[0])
                            if score >= float(config.EMERGENCY_LIGHT_MIN_PIXELS):
                                if emergency_light_prev_score is not None and emergency_light_prev_score > 0:
                                    change = abs(score - emergency_light_prev_score) / emergency_light_prev_score
                                else:
                                    change = 1.0
                                if change >= float(config.EMERGENCY_LIGHT_CHANGE_RATIO):
                                    emergency_confirmed = True
                            emergency_light_prev_score = score

                if emergency_confirmed:
                    emergency_confirmed_until = now + float(config.EMERGENCY_HOLD_SECONDS)

            # Update controller (handles GREEN/YELLOW switching and fairness),
            # unless emergency override is active.
            if (
                config.EMERGENCY_MODE
                and emergency_direction
                and (now < emergency_confirmed_until)
            ):
                # Emergency override is intentionally immediate for safety.
                controller.force_green(emergency_direction)
            else:
                controller.update(density)

            # Draw bounding boxes.
            for (x1, y1, x2, y2), label in zip(boxes, labels):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 0),
                    2,
                )

            # Build a clean overlay panel.
            panel_x, panel_y = 10, 10
            panel_w, panel_h = 460, 130
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)

            # Overlay: counts + per-region.
            cv2.putText(
                frame,
                f"Vehicles: {vehicle_count} | Regions N:{region_counts['north']} S:{region_counts['south']} E:{region_counts['east']} W:{region_counts['west']}",
                (panel_x + 10, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Emergency status line.
            if config.EMERGENCY_MODE:
                if emergency_direction and now < emergency_confirmed_until:
                    em_text = f"EMERGENCY ACTIVE -> {emergency_direction.upper()}"
                    em_color = (0, 0, 255)
                elif emergency_direction:
                    em_text = f"Emergency detected ({emergency_direction.upper()}) - waiting confirm (press '{config.EMERGENCY_MANUAL_KEY}')"
                    em_color = (0, 165, 255)
                else:
                    em_text = f"Emergency mode ON (press '{config.EMERGENCY_MANUAL_KEY}' to confirm)"
                    em_color = (200, 200, 200)
                cv2.putText(
                    frame,
                    em_text,
                    (panel_x + 10, panel_y + panel_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    em_color,
                    2,
                )

            # FPS + status (small, but useful for demos).
            frames_for_fps += 1
            now = time.monotonic()
            if now - last_fps_t >= 1.0:
                fps = frames_for_fps / max(1e-6, (now - last_fps_t))
                last_fps_t = now
                frames_for_fps = 0
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Infer: 1/{process_every_n} | {status_line}",
                (panel_x + 10, panel_y + panel_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # Determine active signal (GREEN preferred; else YELLOW).
            active_dir = None
            active_color = None
            for d in ("north", "south", "east", "west"):
                if controller.signals[d].color == "GREEN":
                    active_dir, active_color = d, "GREEN"
                    break
            if active_dir is None:
                for d in ("north", "south", "east", "west"):
                    if controller.signals[d].color == "YELLOW":
                        active_dir, active_color = d, "YELLOW"
                        break
            if active_dir is None:
                active_dir, active_color = controller.current_green, controller.signals[controller.current_green].color

            # Countdown timer (best-effort demo timer based on phase changes).
            now = time.monotonic()
            active_key = (active_dir, active_color)
            if last_active != active_key:
                if active_color == "GREEN":
                    phase_ends_at = now + max(0, int(controller.signals[active_dir].green_seconds))
                elif active_color == "YELLOW":
                    phase_ends_at = now + max(0, int(config.YELLOW_TIME))
                else:
                    phase_ends_at = None
                last_active = active_key

            remaining = None
            if phase_ends_at is not None:
                remaining = max(0, int(round(phase_ends_at - now)))

            color_map = {"GREEN": (0, 255, 0), "YELLOW": (0, 255, 255), "RED": (0, 0, 255)}
            active_bgr = color_map.get(active_color, (255, 255, 255))
            timer_text = f"{active_color} {active_dir.upper()}" + (f" | t-{remaining}s" if remaining is not None else "")
            cv2.putText(
                frame,
                f"Active: {timer_text}",
                (panel_x + 10, panel_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                active_bgr,
                2,
            )

            # Show each direction status in a compact list.
            y = panel_y + 80
            for d in ("north", "south", "east", "west"):
                st = controller.signals[d]
                bgr = color_map.get(st.color, (255, 255, 255))
                extra = f"{st.green_seconds:>2}s" if st.color == "GREEN" else ""
                cv2.putText(
                    frame,
                    f"{d.upper():<5} {st.color:<6} {extra}",
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    bgr,
                    2,
                )
                y += 20

            cv2.imshow("AI Traffic Management", frame)

            key = (cv2.waitKey(1) & 0xFF)
            if key == ord("q"):
                break
            if config.EMERGENCY_MODE and key == ord(str(config.EMERGENCY_MANUAL_KEY).lower()):
                # Manual confirmation: hold emergency override briefly.
                manual_emergency_until = time.monotonic() + float(config.EMERGENCY_HOLD_SECONDS)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log("Shutdown complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())

