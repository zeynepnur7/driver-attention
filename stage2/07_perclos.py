import math
import time
from collections import deque
from statistics import median

import cv2 as cv
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

CALIB_SECONDS = 5
THRESHOLD_RATIO = 0.65
MIN_SAMPLES = 30

WINDOW_SECONDS = 30        # PERCLOS penceresi (test için kısa tuttum)
MIN_WINDOW_FILL = 10       # pencerede en az bu kadar veri olmadan skor gösterme
PERCLOS_WARN = 0.15        # başlangıç değerleri, kendi testinle ayarla
PERCLOS_ALARM = 0.30
MICROSLEEP_SECONDS = 1.0   # kesintisiz kapalı kalma süresi


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_points(landmarks, indices, w, h):
    return [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]


def ear(p):
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]))


def put(frame, text, y, color=(255, 255, 255)):
    cv.putText(frame, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


class PerclosWindow:
    """Son WINDOW_SECONDS içindeki kapalı göz süresinin oranı."""

    def __init__(self, window_seconds):
        self.window = window_seconds
        self.items = deque()          # (zaman, dt, kapalı_mı)

    def add(self, t, dt, closed):
        self.items.append((t, dt, closed))
        while self.items and t - self.items[0][0] > self.window:
            self.items.popleft()

    def covered(self):
        return sum(dt for _, dt, _ in self.items)

    def value(self):
        total = self.covered()
        if total <= 0:
            return 0.0
        closed = sum(dt for _, dt, c in self.items if c)
        return closed / total

    def clear(self):
        self.items.clear()


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı")

t0 = time.perf_counter()
last_ts = -1
last_t = None

calibrating = True
calib_start = None
samples = []
threshold = None

perclos = PerclosWindow(WINDOW_SECONDS)
closed_since = None
microsleep_count = 0
in_microsleep = False

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
        dt = 0.0 if last_t is None else min(now - last_t, 0.2)   # uzun duraksamaları kırp
        last_t = now

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
                put(frame, "KALIBRASYON: gozlerini ACIK tut", 30, (0, 255, 255))
                put(frame, f"Kalan: {max(remaining, 0):.1f} sn", 60, (0, 255, 255))
                if remaining <= 0:
                    if len(samples) >= MIN_SAMPLES:
                        threshold = median(samples) * THRESHOLD_RATIO
                        calibrating = False
                        perclos.clear()
                        print(f"Esik: {threshold:.3f}")
                    else:
                        samples.clear()
                        calib_start = None
            else:
                is_closed = avg_ear < threshold
                perclos.add(now, dt, is_closed)

                # microsleep: kesintisiz kapalı kalma süresi
                if is_closed:
                    if closed_since is None:
                        closed_since = now
                    closed_dur = now - closed_since
                    if closed_dur >= MICROSLEEP_SECONDS and not in_microsleep:
                        in_microsleep = True
                        microsleep_count += 1       # olay başına bir kez say
                else:
                    closed_since = None
                    closed_dur = 0.0
                    in_microsleep = False

                p = perclos.value()
                ready = perclos.covered() >= MIN_WINDOW_FILL

                put(frame, f"EAR: {avg_ear:.3f}  Esik: {threshold:.3f}", 30,
                    (0, 0, 255) if is_closed else (0, 255, 0))
                put(frame, f"Kapali sure: {closed_dur:.1f} sn", 60)

                if ready:
                    color = (0, 255, 0)
                    label = "NORMAL"
                    if p >= PERCLOS_ALARM:
                        color, label = (0, 0, 255), "YORGUN"
                    elif p >= PERCLOS_WARN:
                        color, label = (0, 165, 255), "DIKKAT"
                    put(frame, f"PERCLOS: {p:.2f}  [{label}]", 90, color)
                else:
                    put(frame, f"PERCLOS: olcum dolunuyor "
                               f"({perclos.covered():.0f}/{MIN_WINDOW_FILL} sn)", 90,
                        (200, 200, 200))

                put(frame, f"Microsleep: {microsleep_count}", 120,
                    (0, 0, 255) if in_microsleep else (255, 255, 255))
                if in_microsleep:
                    put(frame, "UYARI: MICROSLEEP", 160, (0, 0, 255))
        else:
            # yüz yok: veri ekleme, kapanma sayacını sıfırla
            closed_since = None
            in_microsleep = False
            put(frame, "Yuz bulunamadi", 30, (0, 0, 255))

        put(frame, "r: yeniden kalibre et  q: cikis", h - 15, (200, 200, 200))
        cv.imshow("PERCLOS", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            calibrating = True
            calib_start = None
            samples.clear()
            threshold = None
            perclos.clear()
            closed_since = None
            microsleep_count = 0
            in_microsleep = False

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)