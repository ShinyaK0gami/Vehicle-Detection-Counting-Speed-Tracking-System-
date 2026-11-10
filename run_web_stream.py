cat << 'EOF' > run_web_stream.py
#!/usr/bin/env python3
import time
import math
import threading
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from flask import Flask, Response, render_template_string

# ==================== CONFIGURATION ====================
ENGINE_PATH = "best.engine"
VIDEO_SOURCE = "Test_Video.mp4"
CONF_THRESH = 0.35
IOU_THRESH = 0.45
INPUT_SIZE = 416
WEB_PORT = 5000

CLASSES = ["car", "truck", "trolley", "bus", "motorbike", "van"]

CLASS_COLORS = {
    "car":       (0, 255, 0),     # Bright Green
    "truck":     (0, 140, 255),   # Orange
    "trolley":   (255, 255, 0),   # Cyan
    "bus":       (255, 60, 0),    # Deep Blue
    "motorbike": (0, 255, 255),   # Bright Yellow
    "van":       (220, 0, 255)    # Magenta
}

LINE_Y_RATIO = 0.65
# =======================================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Global buffer for threaded web streaming
output_frame = None
lock = threading.Lock()

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

                # Perspective-calibrated velocity
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

                # Tripwire Counting
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
    global output_frame, lock

    # Initialize PyCUDA context on this thread
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

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error opening video source:", VIDEO_SOURCE)
        cuda_ctx.pop()
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 1.0 or math.isnan(src_fps):
        src_fps = 30.0
    dt_per_frame = 1.0 / src_fps

    tracker = VehicleTracker(max_lost=6, max_distance=65)
    fps_smoothed = 0.0
    prev_time = time.perf_counter()
    frame_idx = 0

    print("Background TensorRT worker active. Streaming ready.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

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

        out = outputs[0]["host"].reshape((4 + len(CLASSES), -1)).T
        boxes, confidences, class_ids = [], [], []

        for row in out:
            classes_scores = row[4:]
            cls_id = np.argmax(classes_scores)
            score = classes_scores[cls_id]
            if score > CONF_THRESH:
                cx, cy, w, h = row[0:4]
                x1 = int((cx - w / 2) / ratio)
                y1 = int((cy - h / 2) / ratio)
                bw = int(w / ratio)
                bh = int(h / ratio)
                boxes.append([x1, y1, bw, bh])
                confidences.append(float(score))
                class_ids.append(cls_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, IOU_THRESH)
        filtered_dets = []
        if len(indices) > 0:
            for i in indices.flatten():
                filtered_dets.append((boxes[i], class_ids[i], confidences[i]))

        tracked_objects = tracker.update(filtered_dets, line_y, frame_idx, dt_per_frame, frame_h)

        # 1. Tripwire
        cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "TRIPWIRE", (15, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        # 2. Vehicle Boxes & Clean Labels
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

        # 3. Performance HUD
        loop_time = curr_time - prev_time
        prev_time = curr_time
        current_fps = 1.0 / max(loop_time, 1e-5)
        fps_smoothed = current_fps if fps_smoothed == 0.0 else (0.9 * fps_smoothed + 0.1 * current_fps)

        cv2.rectangle(frame, (10, 10), (180, 65), (20, 20, 20), -1)
        cv2.putText(frame, "FPS: {:.1f}".format(fps_smoothed), (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "GPU: {:.1f} ms".format(gpu_ms), (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Vehicle Counts Scoreboard
        panel_w = 160
        cv2.rectangle(frame, (frame_w - panel_w - 10, 10), (frame_w - 10, 30 + len(CLASSES) * 22), (20, 20, 20), -1)
        cv2.putText(frame, "COUNTS", (frame_w - panel_w + 5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        for idx, cls_name in enumerate(CLASSES):
            col = CLASS_COLORS[cls_name]
            cnt = tracker.counts[cls_name]
            y_pos = 50 + idx * 22
            cv2.putText(frame, "{}:".format(cls_name.capitalize()), (frame_w - panel_w + 5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            cv2.putText(frame, str(cnt), (frame_w - 38, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Encode to JPEG for web streaming
        ret_enc, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ret_enc:
            with lock:
                output_frame = jpeg.tobytes()

    cap.release()
    cuda_ctx.pop()

# ==================== FLASK SERVER ====================
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
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

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

def generate_stream():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            frame_bytes = output_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.02)

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=inference_worker)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True)
EOF
