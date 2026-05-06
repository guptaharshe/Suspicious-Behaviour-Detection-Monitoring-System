"""
Loitering Detection System
===========================
A single-script loitering detection system using:
  - YOLOv8n for person detection
  - DeepSORT for multi-person tracking with persistent IDs
  - Configurable two-tier thresholds (presence + loitering)
  - Annotated video output (MP4) with console/text event logging

Usage:
    python loitering_detector.py --input path/to/video.mp4
    python loitering_detector.py --input path/to/video.mp4 --output result.mp4 --log events.txt
    python loitering_detector.py --input path/to/video.mp4 --presence-threshold 6.3 --loitering-threshold 8.1
"""

import argparse
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ──────────────────────────────────────────────────────────────────────────────
# Configuration defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PRESENCE_THRESHOLD = 10.0   # seconds — show "ID {n} | {t}s" label
DEFAULT_LOITERING_THRESHOLD = 30.0  # seconds — switch to RED + "⚠ Loitering"
DEFAULT_DETECT_EVERY_N = 2          # run YOLO on every Nth frame for speed
DEFAULT_CONFIDENCE = 0.5            # minimum detection confidence
DEFAULT_MAX_AGE = 30                # DeepSORT: max frames to keep lost track
YOLO_MODEL_NAME = "yolov8n.pt"      # lightweight model, CPU-friendly
PERSON_CLASS_ID = 0                 # COCO class ID for "person"

# Visual styling
COLOR_GREEN = (118, 230, 0)         # normal presence bounding box (BGR)
COLOR_RED = (53, 57, 229)           # loitering alert bounding box (BGR #E53935)
COLOR_YELLOW = (0, 255, 255)        # presence label text
COLOR_WHITE = (255, 255, 255)       # general text
COLOR_BG_DARK = (20, 20, 20)        # label background
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 3                   # thick bounding boxes
BOX_RADIUS = 12                     # rounded corner radius


# ──────────────────────────────────────────────────────────────────────────────
# State Tracker — manages per-ID timing and loitering status
# ──────────────────────────────────────────────────────────────────────────────

class PersonState:
    """Stores the state for a single tracked person."""

    def __init__(self, track_id: int, first_seen: float):
        self.track_id = track_id
        self.first_seen = first_seen      # timestamp when first detected
        self.last_seen = first_seen       # timestamp of most recent detection
        self.loitering = False            # whether loitering alert is active
        self.loitering_logged = False     # whether "loitering started" was logged
        self.entered_logged = False       # whether "entered" was logged

    @property
    def duration(self) -> float:
        """Duration in seconds since first seen."""
        return self.last_seen - self.first_seen


class StateManager:
    """Manages all tracked person states."""

    def __init__(self):
        self.states: OrderedDict[int, PersonState] = OrderedDict()

    def update(self, track_id: int, current_time: float) -> PersonState:
        """Update or create state for a tracked person."""
        if track_id not in self.states:
            self.states[track_id] = PersonState(track_id, current_time)
        state = self.states[track_id]
        state.last_seen = current_time
        return state

    def cleanup(self, active_ids: set, max_missing_seconds: float = 5.0, 
                current_time: float = 0.0):
        """Remove states for IDs that are no longer being tracked."""
        stale_ids = [
            tid for tid, state in self.states.items()
            if tid not in active_ids 
            and (current_time - state.last_seen) > max_missing_seconds
        ]
        for tid in stale_ids:
            del self.states[tid]


# ──────────────────────────────────────────────────────────────────────────────
# Event Logger — console + optional file logging
# ──────────────────────────────────────────────────────────────────────────────

class EventLogger:
    """Logs loitering events to console and optionally to a text file."""

    def __init__(self, log_file: str = None):
        self.log_file = None
        if log_file:
            self.log_file = open(log_file, "w", encoding="utf-8")
            self.log_file.write("Loitering Detection Event Log\n")
            self.log_file.write("=" * 50 + "\n\n")

    def log(self, video_time_sec: float, message: str):
        """Log an event with a video timestamp."""
        minutes = int(video_time_sec) // 60
        seconds = int(video_time_sec) % 60
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        line = f"{timestamp} {message}"
        print(line)
        if self.log_file:
            self.log_file.write(line + "\n")
            self.log_file.flush()

    def close(self):
        """Close the log file if open."""
        if self.log_file:
            self.log_file.close()


# ──────────────────────────────────────────────────────────────────────────────
# Frame Annotator — draws bounding boxes, labels, and alerts on frames
# ──────────────────────────────────────────────────────────────────────────────

def draw_rounded_rect(img, pt1, pt2, color, thickness, radius):
    """Draw a rectangle with rounded corners using lines and ellipse arcs."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    if r <= 0:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return

    # Four straight edges (excluding corners)
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

    # Four corner arcs
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)


def draw_zone(frame, zone_points_px):
    """Draw a monitoring zone polygon on the frame with semi-transparent fill."""
    if not zone_points_px or len(zone_points_px) < 3:
        return

    pts = np.array(zone_points_px, dtype=np.int32)
    zone_color = (212, 188, 0)  # teal in BGR (#00BCD4)

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (212, 188, 0))
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    # Border
    cv2.polylines(frame, [pts], isClosed=True, color=zone_color, thickness=2)

    # Label
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.min(pts[:, 1])) - 10
    cv2.putText(frame, "MONITORING ZONE", (cx - 70, max(cy, 20)),
                FONT, 0.5, zone_color, 1, cv2.LINE_AA)


def is_inside_zone(centroid, zone_points_px):
    """Check if a centroid (x, y) is inside the zone polygon."""
    if not zone_points_px or len(zone_points_px) < 3:
        return True  # no zone = everywhere is monitored
    pts = np.array(zone_points_px, dtype=np.float32)
    result = cv2.pointPolygonTest(pts, (float(centroid[0]), float(centroid[1])), False)
    return result >= 0

def draw_label(frame, text: str, position: tuple, bg_color: tuple, 
               text_color: tuple = COLOR_WHITE):
    """Draw a text label with a filled background rectangle."""
    x, y = position
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, 
                                                  FONT_THICKNESS)
    # Background rectangle with slight rounding
    cv2.rectangle(frame, 
                  (x, y - text_h - baseline - 8),
                  (x + text_w + 10, y + 2),
                  bg_color, -1)
    # Text
    cv2.putText(frame, text, (x + 5, y - baseline - 2), FONT, FONT_SCALE,
                text_color, FONT_THICKNESS, cv2.LINE_AA)


def annotate_frame(frame, bbox: tuple, state: PersonState, 
                   presence_threshold: float, loitering_threshold: float):
    """
    Annotate a single tracked person on the frame.
    
    - Green rounded box + ID label when duration >= presence_threshold
    - Red rounded box + loitering warning when duration >= loitering_threshold
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    duration = state.duration

    if duration >= loitering_threshold:
        # ── LOITERING ALERT ──
        state.loitering = True
        draw_rounded_rect(frame, (x1, y1), (x2, y2), COLOR_RED,
                          BOX_THICKNESS + 1, BOX_RADIUS)

        # Warning label
        alert_text = f"!! Loitering | {duration:.1f}s"
        draw_label(frame, alert_text, (x1, y1 - 6), COLOR_RED, COLOR_WHITE)

        # Pulsing red glow effect around the box
        overlay = frame.copy()
        draw_rounded_rect(overlay, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4),
                          COLOR_RED, 2, BOX_RADIUS + 4)
        alpha = 0.4 + 0.2 * abs(np.sin(time.time() * 4))
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    elif duration >= presence_threshold:
        # ── NORMAL PRESENCE ──
        draw_rounded_rect(frame, (x1, y1), (x2, y2), COLOR_GREEN,
                          BOX_THICKNESS, BOX_RADIUS)

        # ID + duration label
        label_text = f"Person {state.track_id} | {duration:.1f}s"
        draw_label(frame, label_text, (x1, y1 - 6), COLOR_BG_DARK, 
                   COLOR_YELLOW)

    else:
        # ── BRIEF DETECTION (below presence threshold) ──
        draw_rounded_rect(frame, (x1, y1), (x2, y2), (100, 100, 100),
                          1, BOX_RADIUS)


def draw_hud(frame, frame_num: int, fps: float, total_tracked: int, 
             active_alerts: int):
    """Draw a heads-up display overlay with stats."""
    h, w = frame.shape[:2]

    # Semi-transparent bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Stats text
    stats = (f"Frame: {frame_num}  |  FPS: {fps:.1f}  |  "
             f"Tracked: {total_tracked}  |  Alerts: {active_alerts}")
    cv2.putText(frame, stats, (10, 25), FONT, 0.5, COLOR_WHITE, 1, 
                cv2.LINE_AA)

    # Alert indicator dot (pulsing red if there are active alerts)
    if active_alerts > 0:
        radius = int(6 + 2 * abs(np.sin(time.time() * 5)))
        cv2.circle(frame, (w - 25, 18), radius, COLOR_RED, -1)


# ──────────────────────────────────────────────────────────────────────────────
# Main Detection Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_loitering_detection(args):
    """Main pipeline: read video → detect → track → annotate → save."""

    # ── Validate input ──
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # ── Phase 1: Open video ──
    print(f"\n{'=' * 60}")
    print(f"  LOITERING DETECTION SYSTEM")
    print(f"{'=' * 60}")
    print(f"  Input:               {args.input}")
    print(f"  Presence threshold:  {args.presence_threshold}s")
    print(f"  Loitering threshold: {args.loitering_threshold}s")
    print(f"  Detect every N:      {args.detect_every_n} frames")
    print(f"  Confidence:          {args.confidence}")
    print(f"{'=' * 60}\n")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.input}")
        sys.exit(1)

    # Video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video: {frame_width}x{frame_height} @ {original_fps:.1f} FPS "
          f"({total_frames} frames)")

    # ── Output video writer ──
    output_path = args.output
    if not output_path:
        base, _ = os.path.splitext(args.input)
        output_path = f"{base}_loitering_output.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, original_fps,
                             (frame_width, frame_height))
    print(f"  Output: {output_path}\n")

    # ── Phase 2: Initialize YOLOv8 ──
    print("  Loading YOLOv8n model...")
    model = YOLO(YOLO_MODEL_NAME)
    print("  Model loaded successfully.\n")

    # ── Phase 3: Initialize DeepSORT ──
    tracker = DeepSort(
        max_age=args.max_age,
        n_init=3,                   # confirmations needed before track is active
        max_iou_distance=0.7,
        embedder="mobilenet",       # lightweight appearance embedder
        half=False,                 # full precision for CPU
        bgr=True,                   # OpenCV frames are BGR
    )

    # ── Initialize state manager and logger ──
    state_manager = StateManager()
    logger = EventLogger(args.log)

    # ── Processing loop ──
    frame_num = 0
    last_detections = []            # cache detections between YOLO runs
    processing_times = []           # for FPS calculation
    total_unique_tracked = set()

    print("  Processing video...\n")
    logger.log(0, "Detection started")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            frame_start = time.time()

            # Video timestamp in seconds
            video_time = frame_num / original_fps

            # ── Phase 2: Person detection (every Nth frame) ──
            if frame_num % args.detect_every_n == 0 or frame_num == 1:
                results = model(frame, verbose=False, conf=args.confidence)
                detections = []

                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        if cls_id != PERSON_CLASS_ID:
                            continue
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        w = x2 - x1
                        h = y2 - y1
                        # DeepSORT expects: ([x, y, w, h], confidence, class_name)
                        detections.append(
                            ([float(x1), float(y1), float(w), float(h)], 
                             conf, "person")
                        )

                last_detections = detections
            else:
                detections = last_detections

            # ── Phase 3: Update DeepSORT tracker ──
            tracks = tracker.update_tracks(detections, frame=frame)

            active_ids = set()
            active_alerts = 0

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_ltrb()  # (left, top, right, bottom)

                # Validate bounding box
                if any(v < 0 for v in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue

                active_ids.add(track_id)
                total_unique_tracked.add(track_id)

                # ── Phase 4: Update state and apply loitering logic ──
                state = state_manager.update(track_id, video_time)

                # Log "entered" event
                if not state.entered_logged:
                    logger.log(video_time, f"Person {track_id} entered")
                    state.entered_logged = True

                # Check loitering threshold
                if (state.duration >= args.loitering_threshold 
                        and not state.loitering_logged):
                    logger.log(video_time, 
                               f"Person {track_id} loitering started "
                               f"(duration: {state.duration:.1f}s)")
                    state.loitering_logged = True
                    state.loitering = True

                # Periodic logging for ongoing loitering (every 5 seconds)
                if (state.loitering 
                        and int(state.duration) % 5 == 0 
                        and int(state.duration) > int(args.loitering_threshold)):
                    # Only log once per 5-second mark
                    marker = int(state.duration)
                    if not hasattr(state, '_last_periodic_log') or state._last_periodic_log != marker:
                        logger.log(video_time, 
                                   f"Person {track_id} still loitering "
                                   f"({state.duration:.1f}s)")
                        state._last_periodic_log = marker

                if state.loitering:
                    active_alerts += 1

                # ── Phase 5: Annotate the frame ──
                annotate_frame(frame, bbox, state, 
                               args.presence_threshold, 
                               args.loitering_threshold)

            # Cleanup stale states
            state_manager.cleanup(active_ids, max_missing_seconds=3.0,
                                  current_time=video_time)

            # ── Draw HUD ──
            frame_time = time.time() - frame_start
            processing_times.append(frame_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
            avg_fps = 1.0 / (sum(processing_times) / len(processing_times)) \
                if processing_times else 0

            draw_hud(frame, frame_num, avg_fps, len(active_ids), active_alerts)

            # ── Write output frame ──
            writer.write(frame)

            # ── Show live preview (optional, press 'q' to quit) ──
            if not args.no_display:
                # Resize for display if frame is too large
                display_frame = frame
                if frame_width > 1280:
                    scale = 1280 / frame_width
                    display_frame = cv2.resize(
                        frame, None, fx=scale, fy=scale, 
                        interpolation=cv2.INTER_AREA
                    )
                cv2.imshow("Loitering Detection", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  [!] Stopped by user (pressed 'q')")
                    break

            # Progress indicator
            if frame_num % 100 == 0:
                progress = (frame_num / total_frames) * 100 if total_frames > 0 else 0
                print(f"  Progress: {frame_num}/{total_frames} frames "
                      f"({progress:.1f}%) — {avg_fps:.1f} FPS")

    except KeyboardInterrupt:
        print("\n  [!] Interrupted by user")

    finally:
        # ── Cleanup ──
        cap.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        # ── Final summary ──
        logger.log(video_time if 'video_time' in dir() else 0, "Detection ended")

        print(f"\n{'=' * 60}")
        print(f"  DETECTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Frames processed:     {frame_num}")
        print(f"  Unique persons tracked: {len(total_unique_tracked)}")
        print(f"  Output saved to:      {output_path}")
        if args.log:
            print(f"  Event log saved to:   {args.log}")
        print(f"{'=' * 60}\n")

        logger.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Loitering Detection System — "
                    "Detects persons lingering in video footage using "
                    "YOLOv8 + DeepSORT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python loitering_detector.py --input video.mp4
  python loitering_detector.py --input video.mp4 --output result.mp4 --log events.txt
  python loitering_detector.py --input video.mp4 --presence-threshold 10 --loitering-threshold 15
  python loitering_detector.py --input video.mp4 --no-display
        """
    )

    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input video file (MP4, AVI, MOV)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to output annotated video (default: <input>_loitering_output.mp4)"
    )
    parser.add_argument(
        "--log", "-l", type=str, default=None,
        help="Path to save event log as a text file (optional)"
    )
    parser.add_argument(
        "--presence-threshold", type=float, 
        default=DEFAULT_PRESENCE_THRESHOLD,
        help=f"Seconds before showing presence label (default: {DEFAULT_PRESENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--loitering-threshold", type=float, 
        default=DEFAULT_LOITERING_THRESHOLD,
        help=f"Seconds before triggering loitering alert (default: {DEFAULT_LOITERING_THRESHOLD})"
    )
    parser.add_argument(
        "--detect-every-n", type=int, default=DEFAULT_DETECT_EVERY_N,
        help=f"Run YOLO detection every Nth frame (default: {DEFAULT_DETECT_EVERY_N})"
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help=f"Minimum detection confidence (default: {DEFAULT_CONFIDENCE})"
    )
    parser.add_argument(
        "--max-age", type=int, default=DEFAULT_MAX_AGE,
        help=f"DeepSORT max age for lost tracks (default: {DEFAULT_MAX_AGE})"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable live preview window (headless processing)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_loitering_detection(args)
