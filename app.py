import os
import threading
import time
from typing import Generator, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, jsonify, send_from_directory


app = Flask(__name__, static_url_path='/static', static_folder='static')


class PeopleCounterStream:
    def __init__(self, source: int | str = 0, model_path: str = 'yolov8s.pt', frame_width: int = 1020, frame_height: int = 600):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        self.model = YOLO(model_path)
        self.frame_lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.running = False
        self.enabled = True

        # Counters
        self.total_count = 0  # total unique IDs observed

        # Heatmap accumulator (grayscale float32)
        self.heatmap_accum: np.ndarray | None = None

        # Accuracy (confidence) exponential moving average
        self.confidence_ema: float = 0.0
        self.confidence_alpha: float = 0.2

        # Crossing lines
        self.cy1 = int(frame_height * 0.4)
        self.cy2 = int(frame_height * 0.48)
        self.offset = 8

        # Tracker state: id -> last center point
        self.next_id = 0
        self.id_to_center: dict[int, Tuple[int, int]] = {}
        self.assigned_ids: set[int] = set()

    def _update_tracker(self, boxes: list[Tuple[int, int, int, int]]) -> list[Tuple[int, int, int, int, int]]:
        objects_bbs_ids: list[Tuple[int, int, int, int, int]] = []
        new_centers: dict[int, Tuple[int, int]] = {}

        for (x1, y1, x2, y2) in boxes:
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            assigned_id = None
            # simple nearest neighbor
            for obj_id, (px, py) in self.id_to_center.items():
                if (cx - px) ** 2 + (cy - py) ** 2 < 35 ** 2:
                    assigned_id = obj_id
                    break
            if assigned_id is None:
                assigned_id = self.next_id
                self.next_id += 1
                # New unique object observed
                if assigned_id not in self.assigned_ids:
                    self.assigned_ids.add(assigned_id)
                    self.total_count += 1
            new_centers[assigned_id] = (cx, cy)
            objects_bbs_ids.append((x1, y1, x2, y2, assigned_id))

        self.id_to_center = new_centers
        return objects_bbs_ids

    def _update_heatmap(self, frame_shape: Tuple[int, int, int], centers: list[Tuple[int, int]]):
        h, w = frame_shape[:2]
        if self.heatmap_accum is None or self.heatmap_accum.shape[:2] != (h, w):
            self.heatmap_accum = np.zeros((h, w), dtype=np.float32)
        for (cx, cy) in centers:
            cv2.circle(self.heatmap_accum, (int(cx), int(cy)), 15, 1.0, -1)
        # gentle decay
        self.heatmap_accum *= 0.97

    def _overlay_heatmap(self, frame: np.ndarray) -> np.ndarray:
        if self.heatmap_accum is None:
            return frame
        hm = self.heatmap_accum.copy()
        hm = np.clip(hm / (hm.max() + 1e-6), 0, 1)
        hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.7, hm_color, 0.3, 0)
        return overlay

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Inference
        results = self.model.predict(frame, verbose=False)
        det = results[0]
        boxes_xyxy = det.boxes.xyxy.cpu().numpy() if hasattr(det.boxes, 'xyxy') else np.empty((0, 4))
        cls = det.boxes.cls.cpu().numpy().astype(int) if hasattr(det.boxes, 'cls') else np.empty((0,), dtype=int)
        conf = det.boxes.conf.cpu().numpy() if hasattr(det.boxes, 'conf') else np.empty((0,), dtype=float)

        person_boxes: list[Tuple[int, int, int, int]] = []
        person_confidences: list[float] = []
        for i in range(len(boxes_xyxy)):
            if i < len(cls) and cls[i] == 0:  # COCO class 0 is person
                x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
                person_boxes.append((x1, y1, x2, y2))
                if i < len(conf):
                    person_confidences.append(float(conf[i]))

        tracked = self._update_tracker(person_boxes)

        centers = []
        # Draw detections and collect centers
        for (x1, y1, x2, y2, oid) in tracked:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centers.append((cx, cy))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
            cv2.putText(frame, f"{oid}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # Update heatmap accumulator
        self._update_heatmap(frame.shape, centers)

        # Overlay total count
        # Update accuracy EMA
        if len(person_confidences) > 0:
            batch_conf = float(np.mean(person_confidences))
            self.confidence_ema = (1 - self.confidence_alpha) * self.confidence_ema + self.confidence_alpha * batch_conf
        acc_pct = int(round(self.confidence_ema * 100))

        cv2.putText(frame, f"Total: {self.total_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Accuracy: {acc_pct}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

        # Overlay heatmap visualization
        frame = self._overlay_heatmap(frame)
        return frame

    def frames(self) -> Generator[bytes, None, None]:
        self.running = True
        try:
            while self.running and self.cap.isOpened():
                if not self.enabled:
                    # yield a paused frame to keep connection alive
                    paused = np.zeros((600, 1020, 3), dtype=np.uint8)
                    cv2.putText(paused, "STREAM PAUSED", (330, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
                    ret, jpeg = cv2.imencode('.jpg', paused, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ret:
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                    time.sleep(0.2)
                    continue
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                frame = cv2.resize(frame, (1020, 600))
                frame = self._process_frame(frame)
                ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret:
                    continue
                with self.frame_lock:
                    self.latest_jpeg = jpeg.tobytes()
                    payload = self.latest_jpeg
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n")
        finally:
            self.running = False
            self.cap.release()

    def last_frame(self) -> bytes | None:
        with self.frame_lock:
            return self.latest_jpeg

    def get_counts(self) -> tuple[int, int, int]:
        current = len(self.id_to_center)
        acc_pct = int(round(self.confidence_ema * 100))
        return self.total_count, current, acc_pct

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled


streamer = PeopleCounterStream(source=0, model_path='yolov8s.pt')


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/stream')
def stream():
    return Response(streamer.frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/counts')
def counts():
    total, current, accuracy = streamer.get_counts()
    return jsonify({"total": total, "current": current, "accuracy": accuracy})


@app.route('/stream/start', methods=['POST'])
def start_stream():
    streamer.set_enabled(True)
    return jsonify({"ok": True, "enabled": True})


@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    streamer.set_enabled(False)
    return jsonify({"ok": True, "enabled": False})


@app.route('/static/<path:path>')
def static_files(path: str):
    return send_from_directory('static', path)


def main():
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


if __name__ == '__main__':
    main()


