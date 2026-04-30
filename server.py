"""
server.py — Flask + SocketIO backend for Loitering Detection Web UI.

Streams annotated video via MJPEG and pushes real-time events via SocketIO.

Usage:
    python server.py
    python server.py --video test_video.mp4
"""

import os
import sys
import time
import threading
import json
from collections import OrderedDict

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from loitering_detector import (
    PersonState, StateManager,
    draw_rounded_rect, draw_label, annotate_frame, draw_hud,
    draw_zone, is_inside_zone,
    COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_BG_DARK,
    PERSON_CLASS_ID, YOLO_MODEL_NAME
)

# ── Flask app setup ──────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Global detection state ───────────────────────────────────────────────────

class DetectionEngine:
    """Thread-safe detection state shared between the pipeline and Flask."""

    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.video_path = None
        self.presence_threshold = 10.0
        self.loitering_threshold = 30.0
        self.detect_every_n = 1  # Process every frame for maximum tracking accuracy
        self.confidence = 0.35   # Lower confidence to catch partially occluded people
        self.max_age = 90        # Keep track alive for up to 90 frames (3 seconds) if occluded

        # Zone monitoring (normalized 0-1 coordinates)
        self.zone_points = []  # [{x: 0.1, y: 0.2}, ...]

        # Shared frame for MJPEG streaming
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Events and alerts
        self.events = []
        self.active_alerts = {}  # {track_id: {id, duration, message}}
        self.stats = {
            "frame_num": 0,
            "fps": 0.0,
            "total_tracked": 0,
            "active_alerts": 0,
            "progress": 0.0,
            "status": "idle"
        }

        self._stop_flag = threading.Event()
        self._thread = None

    def add_event(self, video_time: float, message: str, event_type: str = "info"):
        """Add an event and emit it via SocketIO."""
        minutes = int(video_time) // 60
        seconds = int(video_time) % 60
        timestamp = f"[{minutes:02d}:{seconds:02d}]"

        event = {
            "timestamp": timestamp,
            "message": message,
            "type": event_type,
            "raw_time": video_time
        }
        self.events.append(event)
        socketio.emit("event", event)

    def set_alert(self, track_id, duration):
        """Set or update a loitering alert (throttled to 1 emit/sec per ID)."""
        now = time.time()
        alert = {
            "id": track_id,
            "duration": round(duration, 1),
            "message": f"LOITERING DETECTED (Person {track_id})"
        }
        is_new = track_id not in self.active_alerts
        self.active_alerts[track_id] = alert

        # Emit immediately for new alerts, throttle updates to 1/sec
        last_key = f"_alert_{track_id}"
        last_emit = getattr(self, last_key, 0)
        if is_new or (now - last_emit) >= 1.0:
            setattr(self, last_key, now)
            socketio.emit("alert", alert)

    def clear_alert(self, track_id):
        """Clear a loitering alert."""
        if track_id in self.active_alerts:
            del self.active_alerts[track_id]
            socketio.emit("alert_clear", {"id": track_id})

    def update_stats(self, **kwargs):
        """Update stats and emit via SocketIO (throttled to ~3/sec)."""
        self.stats.update(kwargs)
        now = time.time()
        if not hasattr(self, '_last_stats_emit'):
            self._last_stats_emit = 0
        if (now - self._last_stats_emit) >= 0.33:
            self._last_stats_emit = now
            socketio.emit("stats", self.stats)


engine = DetectionEngine()


# ── Detection pipeline (runs in background thread) ──────────────────────────

def detection_loop():
    """Main detection pipeline running in a background thread."""
    global engine

    video_path = engine.video_path
    is_webcam = video_path == "0" or video_path == "webcam"
    is_stream = str(video_path).startswith("http://") or str(video_path).startswith("https://") or str(video_path).startswith("rtsp://")

    if not is_webcam and not is_stream and (not video_path or not os.path.isfile(video_path)):
        engine.add_event(0, "Error: Video file not found or invalid stream", "error")
        engine.stats["status"] = "error"
        engine.is_running = False
        return

    source = 0 if is_webcam else video_path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        engine.add_event(0, "Error: Could not open video", "error")
        engine.stats["status"] = "error"
        engine.is_running = False
        return

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize model and tracker
    engine.add_event(0, "Loading YOLOv8n model...", "info")
    engine.stats["status"] = "loading"
    socketio.emit("stats", engine.stats)

    model = YOLO(YOLO_MODEL_NAME)
    tracker = DeepSort(
        max_age=engine.max_age,
        n_init=2,  # Confirm tracks faster
        max_iou_distance=0.7,
        max_cosine_distance=0.4, # Allow more appearance variation to stop ID switching
        embedder="mobilenet",
        half=False,
        bgr=True,
    )

    state_manager = StateManager()
    engine.add_event(0, "Detection started", "info")
    engine.stats["status"] = "running"

    frame_num = 0
    last_detections = []
    processing_times = []
    total_unique = set()

    try:
        while not engine._stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                engine.add_event(0, "Video finished", "info")
                break

            frame_num += 1
            frame_start = time.time()
            video_time = frame_num / fps

            # ── Person detection (every Nth frame) ──
            if frame_num % engine.detect_every_n == 0 or frame_num == 1:
                results = model(frame, verbose=False, conf=engine.confidence)
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
                        w, h = x2 - x1, y2 - y1
                        detections.append(
                            ([float(x1), float(y1), float(w), float(h)],
                             conf, "person")
                        )
                last_detections = detections
            else:
                detections = last_detections

            # ── Update tracker ──
            tracks = tracker.update_tracks(detections, frame=frame)

            # ── Convert zone to pixel coordinates ──
            h, w = frame.shape[:2]
            zone_px = None
            if engine.zone_points and len(engine.zone_points) >= 3:
                zone_px = [[int(p["x"] * w), int(p["y"] * h)]
                           for p in engine.zone_points]

            # ── Draw zone overlay ──
            draw_zone(frame, zone_px)

            active_ids = set()
            alert_count = 0

            for track in tracks:
                if not track.is_confirmed():
                    continue

                # Prevent massive "ghost" bounding boxes when a track is occluded
                if track.time_since_update > 2:
                    continue

                track_id = track.track_id
                bbox = track.to_ltrb()

                if any(v < 0 for v in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                # ── Check if person is inside zone ──
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                bottom_y = bbox[3]
                
                # Zone check: use both bottom-center (feet) and exact center for robustness
                in_zone = is_inside_zone((cx, bottom_y), zone_px) or is_inside_zone((cx, cy), zone_px)

                if not in_zone:
                    engine.clear_alert(track_id)
                    # Draw a faint box for people outside the zone so the user knows they are detected
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    draw_rounded_rect(frame, (x1, y1), (x2, y2), (100, 100, 100), 1, BOX_RADIUS)
                    continue

                active_ids.add(track_id)
                total_unique.add(track_id)

                # ── Loitering & Movement logic ──
                state = state_manager.update(track_id, video_time)

                bbox_w = bbox[2] - bbox[0]
                if not hasattr(state, 'anchor_centroid'):
                    state.anchor_centroid = (cx, cy)
                else:
                    dx = cx - state.anchor_centroid[0]
                    dy = cy - state.anchor_centroid[1]
                    dist = (dx**2 + dy**2)**0.5
                    
                    # If they move significantly from their anchor spot (1.5x their width)
                    if dist > bbox_w * 1.5:
                        state.first_seen = video_time
                        state.anchor_centroid = (cx, cy)
                        if getattr(state, 'loitering', False):
                            state.loitering = False
                            state.loitering_logged = False
                            engine.clear_alert(track_id)
                            engine.add_event(video_time, f"Person {track_id} started moving (Loitering cleared)", "info")

                if not state.entered_logged:
                    engine.add_event(video_time, f"Person {track_id} entered zone", "enter")
                    state.entered_logged = True

                if (state.duration >= engine.loitering_threshold
                        and not state.loitering_logged):
                    engine.add_event(
                        video_time,
                        f"Person {track_id} loitering in zone ({state.duration:.1f}s)",
                        "loitering"
                    )
                    state.loitering_logged = True
                    state.loitering = True

                # Periodic "still loitering" log
                if state.loitering:
                    marker = int(state.duration)
                    if (marker % 5 == 0
                            and marker > int(engine.loitering_threshold)):
                        if not hasattr(state, '_last_periodic') or state._last_periodic != marker:
                            engine.add_event(
                                video_time,
                                f"Person {track_id} still loitering ({state.duration:.1f}s)",
                                "loitering"
                            )
                            state._last_periodic = marker

                if state.loitering:
                    alert_count += 1
                    engine.set_alert(track_id, state.duration)

                # ── Annotate frame ──
                annotate_frame(frame, bbox, state,
                               engine.presence_threshold,
                               engine.loitering_threshold)

            # Clear alerts for people who left
            stale_alerts = [tid for tid in engine.active_alerts if tid not in active_ids]
            for tid in stale_alerts:
                engine.clear_alert(tid)

            state_manager.cleanup(active_ids, max_missing_seconds=3.0,
                                  current_time=video_time)

            # ── HUD ──
            frame_time = time.time() - frame_start
            processing_times.append(frame_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
            avg_fps = 1.0 / (sum(processing_times) / len(processing_times)) \
                if processing_times else 0

            draw_hud(frame, frame_num, avg_fps, len(active_ids), alert_count)

            # ── Store frame for streaming ──
            with engine.frame_lock:
                engine.current_frame = frame.copy()

            # ── Update stats ──
            progress = (frame_num / total_frames * 100) if total_frames > 0 else 0
            engine.update_stats(
                frame_num=frame_num,
                fps=round(avg_fps, 1),
                total_tracked=len(total_unique),
                active_alerts=alert_count,
                progress=round(progress, 1)
            )

            # Throttle to ~real-time playback
            elapsed = time.time() - frame_start
            target = 1.0 / fps
            if elapsed < target:
                time.sleep(target - elapsed)

    except Exception as e:
        engine.add_event(0, f"Error: {str(e)}", "error")

    finally:
        cap.release()
        engine.stats["status"] = "stopped"
        socketio.emit("stats", engine.stats)
        engine.is_running = False
        engine.add_event(0, "Detection stopped", "info")


# ── Flask routes ─────────────────────────────────────────────────────────────

def generate_frames():
    """Generator yielding MJPEG frames for streaming."""
    while True:
        with engine.frame_lock:
            frame = engine.current_frame

        if frame is not None:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")
        else:
            # Send a blank frame if no video is loaded
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No video loaded", (180, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            _, buffer = cv2.imencode(".jpg", blank)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")

        time.sleep(0.03)  # ~30 FPS max


@app.route("/api/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/events")
def get_events():
    """Return all events."""
    return jsonify(engine.events)


@app.route("/api/stats")
def get_stats():
    """Return current stats."""
    return jsonify(engine.stats)


@app.route("/api/alerts")
def get_alerts():
    """Return active alerts."""
    return jsonify(list(engine.active_alerts.values()))


@app.route("/api/zone", methods=["GET", "POST"])
def zone():
    """Get or set the monitoring zone polygon."""
    if request.method == "GET":
        return jsonify({"points": engine.zone_points})

    data = request.get_json()
    engine.zone_points = data.get("points", [])
    return jsonify({"status": "ok", "points": engine.zone_points})


@app.route("/api/config", methods=["GET", "POST"])
def config():
    """Get or update detection configuration."""
    if request.method == "GET":
        return jsonify({
            "presence_threshold": engine.presence_threshold,
            "loitering_threshold": engine.loitering_threshold,
            "detect_every_n": engine.detect_every_n,
            "confidence": engine.confidence,
        })

    data = request.get_json()
    if "presence_threshold" in data:
        engine.presence_threshold = float(data["presence_threshold"])
    if "loitering_threshold" in data:
        engine.loitering_threshold = float(data["loitering_threshold"])
    if "detect_every_n" in data:
        engine.detect_every_n = int(data["detect_every_n"])
    if "confidence" in data:
        engine.confidence = float(data["confidence"])

    return jsonify({"status": "ok"})


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Upload a video file."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    engine.video_path = filepath
    return jsonify({"status": "ok", "filename": file.filename, "path": filepath})


@app.route("/api/videos")
def list_videos():
    """List available video files."""
    videos = []
    # Check uploads folder
    for f in os.listdir(UPLOAD_FOLDER):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            videos.append({"name": f, "path": os.path.join(UPLOAD_FOLDER, f)})
    # Check project root
    root = os.path.dirname(os.path.abspath(__file__))
    for f in os.listdir(root):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            videos.append({"name": f"(local) {f}", "path": os.path.join(root, f)})
    return jsonify(videos)


@app.route("/api/load_video", methods=["POST"])
def load_video():
    """Load the first frame of a selected video to display in the UI before starting."""
    if engine.is_running:
        return jsonify({"error": "Cannot load video while running"}), 400

    data = request.get_json() or {}
    video_path = data.get("video_path")
    if not video_path:
        return jsonify({"error": "No video selected"}), 400

    engine.video_path = video_path

    is_webcam = str(video_path) == "0" or str(video_path).lower() == "webcam"
    
    try:
        source = 0 if is_webcam else int(video_path)
    except ValueError:
        source = video_path
    
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            with engine.frame_lock:
                engine.current_frame = frame
        cap.release()
    else:
        with engine.frame_lock:
            engine.current_frame = None

    return jsonify({"status": "loaded"})



@app.route("/api/start", methods=["POST"])
def start_detection():
    """Start the detection pipeline."""
    if engine.is_running:
        return jsonify({"error": "Already running"}), 400

    data = request.get_json() or {}
    if "video_path" in data:
        engine.video_path = data["video_path"]

    if not engine.video_path:
        return jsonify({"error": "No video selected"}), 400

    engine.is_running = True
    engine.events.clear()
    engine.active_alerts.clear()
    engine._stop_flag.clear()
    engine.stats["status"] = "starting"
    socketio.emit("stats", engine.stats)

    engine._thread = threading.Thread(target=detection_loop, daemon=True)
    engine._thread.start()

    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def stop_detection():
    """Stop the detection pipeline."""
    engine._stop_flag.set()
    engine.stats["status"] = "stopping"
    socketio.emit("stats", engine.stats)
    return jsonify({"status": "stopping"})


@app.route("/api/reset", methods=["POST"])
def reset_engine():
    """Completely reset the engine to a clean slate."""
    if engine.is_running:
        engine._stop_flag.set()
    
    with engine.frame_lock:
        engine.current_frame = None
    
    engine.video_path = None
    engine.events.clear()
    engine.active_alerts.clear()
    
    engine.stats.update({
        "frame_num": 0,
        "fps": 0.0,
        "total_tracked": 0,
        "active_alerts": 0,
        "progress": 0.0,
        "status": "idle"
    })
    socketio.emit("stats", engine.stats)
    return jsonify({"status": "reset"})


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Suspicious Behavior Detection Web Server")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Pre-load a video file")
    parser.add_argument("--port", "-p", type=int, default=5000)
    args = parser.parse_args()

    if args.video:
        engine.video_path = os.path.abspath(args.video)

    print(f"\n  Suspicious Behavior Detection Server starting on http://localhost:{args.port}")
    print(f"  Video: {engine.video_path or 'None (upload via UI)'}\n")
     
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=args.port, debug=False,
                 allow_unsafe_werkzeug=True)
