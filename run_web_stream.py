#!/usr/bin/env python3
import time
import math
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ==================== CONFIGURATION ====================
ENGINE_PATH = "best.engine"
VIDEO_SOURCE = "Test_Video.mp4"
CONF_THRESH = 0.35
IOU_THRESH = 0.45
INPUT_SIZE = 416
WEB_PORT = 5000

CLASSES = ["car", "truck", "trolley", "bus", "motorbike", "van"]

CLASS_COLORS = {
    "car":       (0, 255, 0),
    "truck":     (0, 140, 255),
    "trolley":   (255, 255, 0),
    "bus":       (255, 60, 0),
    "motorbike": (0, 255, 255),
    "van":       (220, 0, 255)
}

LINE_Y_RATIO = 0.65
# =======================================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

raw_frame = None
raw_lock = threading.Lock()
running = True

annotated_frame = None
annot_lock = threading.Lock()

class BackgroundVideoCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 1.0 or math.isnan(self.fps):
            self.fps = 30.0
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        global raw_frame, raw_lock, running
        while running and self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with raw_lock:
                raw_frame = frame
            time.sleep(0.005)

    def read_latest(self):
        with raw_lock:
            if raw_frame is None:
                return False, None
            return True, raw_frame.copy()

    def release(self):
        self.running = False
        self.cap.release()

class VehicleTracker:
    def __init__(self, max_lost=6, max_distance=65):
        self.next_id = 1
        self.objects = {}
        self.max_lost = max_lost
        self.max_distance = max_distance
        self.counts = {cls: 0 for cls in CLASSES}

    def update(self, detections, line_y, frame_idx, dt_per_frame, frame_h):
        new_centroids = []
        for det in detections:
            x, y, w, h = det[0]
            new_centroids.append((int(x + w / 2), int(y + h / 2)))

        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                self._register(det, new_centroids[i], frame_idx)
            return self.objects

        obj_ids = list(self.objects.keys())
        existing_centroids = [self.objects[oid]["centroid"] for oid in obj_ids]

        if len(new_centroids) > 0:
            dist_matrix = np.linalg.norm(
                np.array(existing_centroids)[:, None] - np.array(new_centroids)[None, :], axis=2
            )
            rows = dist_matrix.min(axis=1).argsort()
            cols = dist_matrix.argmin(axis=1)[rows]

            assigned_rows, assigned_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in assigned_rows or c in assigned_cols:
                    continue
                if dist_matrix[r, c] > self.max_distance:
                    continue

                oid = obj_ids[r]
                det = detections[c]
                cent = new_centroids[c]

                obj = self.objects[oid]
                prev_cent = obj["centroid"]
                obj["bbox"] = det[0]
                obj["cls_id"] = det[1]
                obj["lost"] = 0
                obj["centroid"] = cent
                obj["history"].append((cent[0], cent[1], frame_idx))
                if len(obj["history"]) > 8:
                    obj["history"].pop(0)

                if len(obj["history"]) >= 3:
                    p_old = obj["history"][0]
                    p_new = obj["history"][-1]
                    frames_elapsed = p_new[2] - p_old[2]
                    
                    if frames_elapsed > 0:
                        dt = frames_elapsed * dt_per_frame
                        dy = abs(p_new[1] - p_old[1])
                        dx = abs(p_new[0] - p_old[0])
                        disp_px = math.hypot(dx, dy)

                        y_norm = max(0.2, min(1.0, cent[1] / frame_h))
                        meters_per_px = 0.035 / (y_norm ** 1.3)
                        
                        dist_m = disp_px * meters_per_px
                        raw_speed = (dist_m / dt) * 3.6
                        
                        if 10.0 <= raw_speed <= 140.0:
                            obj["speed"] = 0.65 * obj["speed"] + 0.35 * raw_speed if obj["speed"] > 0 else raw_speed

                if not obj["counted"]:
                    if (prev_cent[1] < line_y and cent[1] >= line_y) or (prev_cent[1] > line_y and cent[1] <= line_y):
                        cls_name = CLASSES[obj["cls_id"]]
                        self.counts[cls_name] += 1
                        obj["counted"] = True

                assigned_rows.add(r)
                assigned_cols.add(c)

            for c in range(len(new_centroids)):
                if c not in assigned_cols:
                    self._register(detections[c], new_centroids[c], frame_idx)

            for r in range(len(obj_ids)):
                if r not in assigned_rows:
                    oid = obj_ids[r]
                    self.objects[oid]["lost"] += 1
                    if self.objects[oid]["lost"] > self.max_lost:
                        del self.objects[oid]
        else:
            for oid in list(self.objects.keys()):
                self.objects[oid]["lost"] += 1
                if self.objects[oid]["lost"] > self.max_lost:
                    del self.objects[oid]

        return self.objects

    def _register(self, det, centroid, frame_idx):
        self.objects[self.next_id] = {
            "bbox": det[0],
            "cls_id": det[1],
            "centroid": centroid,
            "lost": 0,
            "history": [(centroid[0], centroid[1], frame_idx)],
            "counted": False,
            "speed": 0.0
        }
        self.next_id += 1

def preprocess(img, target_size=416):
    h, w = img.shape[:2]
    r = min(target_size / h, target_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    canvas = canvas.transpose((2, 0, 1))[::-1]
    canvas = np.ascontiguousarray(canvas, dtype=np.float32) / 255.0
    return canvas, r

def inference_worker():
    global annotated_frame, annot_lock, running

    cuda_ctx = cuda.Device(0).make_context()
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})

    reader = BackgroundVideoCapture(VIDEO_SOURCE)
    dt_per_frame = 1.0 / reader.fps

    tracker = VehicleTracker(max_lost=6, max_distance=65)
    fps_smoothed = 0.0
    prev_time = time.perf_counter()
    frame_idx = 0

    while running:
        ret, frame = reader.read_latest()
        if not ret:
            time.sleep(0.005)
            continue

        frame_idx += 1
        frame_h, frame_w = frame.shape[:2]
        line_y = int(frame_h * LINE_Y_RATIO)
        curr_time = time.perf_counter()

        inp, ratio = preprocess(frame, INPUT_SIZE)
        np.copyto(inputs[0]["host"], inp.ravel())

        t0 = time.perf_counter()
        cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
        stream.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000.0

        # Vectorized post-processing
        out = outputs[0]["host"].reshape((4 + len(CLASSES), -1)).T
        boxes_raw = out[:, :4]
        scores_raw = out[:, 4:]

        class_ids_all = np.argmax(scores_raw, axis=1)
        confidences_all = np.max(scores_raw, axis=1)
        mask = confidences_all > CONF_THRESH

        boxes, confidences, class_ids = [], [], []
        if np.any(mask):
            filtered_boxes = boxes_raw[mask]
            filtered_confs = confidences_all[mask]
            filtered_cls = class_ids_all[mask]

            cx = filtered_boxes[:, 0]
            cy = filtered_boxes[:, 1]
            w = filtered_boxes[:, 2]
            h = filtered_boxes[:, 3]

            x1 = ((cx - w / 2.0) / ratio).astype(np.int32)
            y1 = ((cy - h / 2.0) / ratio).astype(np.int32)
            bw = (w / ratio).astype(np.int32)
            bh = (h / ratio).astype(np.int32)

            boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()
            confidences = filtered_confs.astype(float).tolist()
            class_ids = filtered_cls.tolist()

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, IOU_THRESH)
        filtered_dets = []
        if len(indices) > 0:
            for i in indices.flatten():
                filtered_dets.append((boxes[i], class_ids[i], confidences[i]))

        tracked_objects = tracker.update(filtered_dets, line_y, frame_idx, dt_per_frame, frame_h)

        # Annotations
        cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "TRIPWIRE", (15, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        for oid, obj in tracked_objects.items():
            if obj["lost"] > 0:
                continue
            x, y, w, h = obj["bbox"]
            cls_name = CLASSES[obj["cls_id"]]
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

            speed_str = "{:.0f} km/h".format(obj["speed"]) if obj["speed"] > 5.0 else "-- km/h"
            tag = "{} | {}".format(cls_name.capitalize(), speed_str)

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tag_y = max(th + 6, y - 6)
            cv2.rectangle(frame, (x, tag_y - th - 5), (x + tw + 6, tag_y + 3), (15, 15, 15), -1)
            cv2.putText(frame, tag, (x + 3, tag_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        loop_time = curr_time - prev_time
        prev_time = curr_time
        current_fps = 1.0 / max(loop_time, 1e-5)
        fps_smoothed = current_fps if fps_smoothed == 0.0 else (0.9 * fps_smoothed + 0.1 * current_fps)

        cv2.rectangle(frame, (10, 10), (180, 65), (20, 20, 20), -1)
        cv2.putText(frame, "FPS: {:.1f}".format(fps_smoothed), (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "GPU: {:.1f} ms".format(gpu_ms), (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        panel_w = 160
        cv2.rectangle(frame, (frame_w - panel_w - 10, 10), (frame_w - 10, 30 + len(CLASSES) * 22), (20, 20, 20), -1)
        cv2.putText(frame, "COUNTS", (frame_w - panel_w + 5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        for idx, cls_name in enumerate(CLASSES):
            col = CLASS_COLORS[cls_name]
            cnt = tracker.counts[cls_name]
            y_pos = 50 + idx * 22
            cv2.putText(frame, "{}:".format(cls_name.capitalize()), (frame_w - panel_w + 5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            cv2.putText(frame, str(cnt), (frame_w - 38, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        with annot_lock:
            annotated_frame = frame

    reader.release()
    cuda_ctx.pop()

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <title>Jetson Nano Edge Telemetry</title>
    <style>
        body { background-color: #121212; color: #ffffff; font-family: Arial, sans-serif; text-align: center; margin: 0; padding: 20px; }
        h1 { margin-bottom: 5px; color: #76b900; }
        p { color: #aaaaaa; margin-top: 0; }
        .stream-box { display: inline-block; border: 2px solid #333333; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        img { display: block; max-width: 90vw; max-height: 80vh; height: auto; }
    </style>
</head>
<body>
    <h1>Jetson Nano - Edge Vehicle Analytics</h1>
    <p>Direct Ethernet Web Stream | YOLOv8 TensorRT FP16</p>
    <div class="stream-box">
        <img src="/video_feed" />
    </div>
</body>
</html>
"""

class StreamingHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        global annotated_frame, annot_lock
        if self.path == '/':
            encoded = HTML_PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        elif self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with annot_lock:
                        target = None if annotated_frame is None else annotated_frame.copy()

                    if target is None:
                        time.sleep(0.01)
                        continue

                    stream_img = cv2.resize(target, (960, 540), interpolation=cv2.INTER_NEAREST)
                    ret_enc, jpeg = cv2.imencode('.jpg', stream_img, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                    if not ret_enc:
                        continue

                    frame_bytes = jpeg.tobytes()
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.02)
            except (ConnectionResetError, BrokenPipeError):
                pass
        else:
            self.send_error(404)
            self.end_headers()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

if __name__ == '__main__':
    t = threading.Thread(target=inference_worker)
    t.daemon = True
    t.start()
    
    server = ThreadedHTTPServer(('0.0.0.0', WEB_PORT), StreamingHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        running = False
        server.server_close()
