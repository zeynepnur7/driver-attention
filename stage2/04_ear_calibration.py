import math
import time
from statistics import median

import cv2 as cv
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

CALIB_SECONDS = 5        # calibration time
THRESHOLD_RATIO = 0.65   # threshold = open eye median * this ratio
MIN_SAMPLES = 30         
MIN_CLOSED_FRAMES = 2    # closed frames for blink count


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_points(landmarks, indices, w, h):
    return [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]


def ear(p):
    vertical = dist(p[1], p[5]) + dist(p[2], p[4])
    horizontal = 2.0 * dist(p[0], p[3])
    return vertical / horizontal


def put(frame, text, y, color=(255, 255, 255)):
    cv.putText(frame, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera can't open")

t0 = time.perf_counter()
last_ts = -1

calibrating = True
calib_start = None
samples = []
threshold = None
closed_frames = 0
blink_count = 0

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        now = time.perf_counter()
        ts = max(last_ts + 1, int((now - t0) * 1000))  
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        if result.face_landmarks:
            lms = result.face_landmarks[0]
            right = eye_points(lms, RIGHT_EYE, w, h)
            left = eye_points(lms, LEFT_EYE, w, h)
            avg_ear = (ear(right) + ear(left)) / 2.0

            for x, y in right + left:
                cv.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            if calibrating:
                if calib_start is None:
                    calib_start = now
                samples.append(avg_ear)
                remaining = CALIB_SECONDS - (now - calib_start)

                put(frame, "CALIBRATION: keep your eyes OPEN", 30, (0, 255, 255))
                put(frame, f"Remaining: {max(remaining, 0):.1f} sec", 60, (0, 255, 255))
                put(frame, f"EAR: {avg_ear:.3f}", 90)

                if remaining <= 0:
                    if len(samples) >= MIN_SAMPLES:
                        open_ear = median(samples)
                        threshold = open_ear * THRESHOLD_RATIO
                        calibrating = False
                        closed_frames = 0
                        blink_count = 0
                        print(f"Open eye EAR (median): {open_ear:.3f}")
                        print(f"Threshold: {threshold:.3f}")
                    else:
                        samples.clear()          # az örnek, baştan başla
                        calib_start = None
            else:
                is_closed = avg_ear < threshold
                if is_closed:
                    closed_frames += 1
                else:
                    if closed_frames >= MIN_CLOSED_FRAMES:
                        blink_count += 1
                    closed_frames = 0

                color = (0, 0, 255) if is_closed else (0, 255, 0)
                put(frame, f"EAR: {avg_ear:.3f}  Threshold: {threshold:.3f}", 30, color)
                put(frame, f"Eye: {'CLOSED' if is_closed else 'OPEN'}", 60, color)
                put(frame, f"Blink: {blink_count}", 90)
                put(frame, "r: recalibrate", h - 15, (200, 200, 200))
        else:
            put(frame, "Face not Found", 30, (0, 0, 255))

        cv.imshow("EAR + Calibration", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            calibrating = True
            calib_start = None
            samples.clear()
            threshold = None

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)