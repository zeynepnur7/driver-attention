import math
import time

import cv2 as cv
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

HEAD_DOWN_PITCH = 15.0     # derece, nötr duruşa göre
LOOK_AWAY_YAW = 25.0       # derece, mutlak değer
HOLD_SECONDS = 1.5         # eşik aşımı bu kadar sürerse durum aktif
SMOOTHING = 0.3            # 0-1, büyükse daha az yumuşatma


def rotation_to_euler(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return tuple(math.degrees(a) for a in (pitch, yaw, roll))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def put(frame, text, y, color=(255, 255, 255)):
    cv.putText(frame, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


class HoldTimer:
    """Koşul kesintisiz ne kadar süredir doğru, onu ölçer."""

    def __init__(self):
        self.since = None

    def update(self, condition, now):
        if condition:
            if self.since is None:
                self.since = now
            return now - self.since
        self.since = None
        return 0.0


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_facial_transformation_matrixes=True,
)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı")

t0 = time.perf_counter()
last_ts = -1
offset = None            # nötr duruş; ilk 'c'ye kadar None
raw = None
smooth = None            # yumuşatılmış (pitch, yaw, roll)
down_timer = HoldTimer()
away_timer = HoldTimer()

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

        mats = result.facial_transformation_matrixes
        if result.face_landmarks and mats:
            raw = rotation_to_euler(np.array(mats[0])[:3, :3])

            if offset is None:
                put(frame, "Duz bak ve 'c'ye bas (notr durus)", 30, (0, 255, 255))
            else:
                cur = tuple(wrap180(raw[i] - offset[i]) for i in range(3))
                if smooth is None:
                    smooth = cur
                else:
                    smooth = tuple(SMOOTHING * cur[i] + (1 - SMOOTHING) * smooth[i]
                                   for i in range(3))
                pitch, yaw, roll = smooth

                down_t = down_timer.update(pitch > HEAD_DOWN_PITCH, now)
                away_t = away_timer.update(abs(yaw) > LOOK_AWAY_YAW, now)

                head_down = down_t >= HOLD_SECONDS
                look_away = away_t >= HOLD_SECONDS

                put(frame, f"Pitch: {pitch:6.1f}  Yaw: {yaw:6.1f}  Roll: {roll:6.1f}", 30)
                put(frame, f"Bas one dusme: {down_t:.1f} sn", 60,
                    (0, 0, 255) if head_down else (255, 255, 255))
                put(frame, f"Yola bakmiyor: {away_t:.1f} sn", 90,
                    (0, 0, 255) if look_away else (255, 255, 255))

                if head_down:
                    put(frame, "WARNING: HEAD DROPPED FORWARD", 130, (0, 0, 255))
                if look_away:
                    put(frame, "WARNING: ATTENTION LOST", 160, (0, 140, 255))
        else:
            # yüz yoksa zamanlayıcıları sıfırla
            down_timer.update(False, now)
            away_timer.update(False, now)
            put(frame, "Face not found", 30, (0, 0, 255))

        put(frame, "c: notr position  q: exit", h - 15, (200, 200, 200))
        cv.imshow("Head State", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and raw is not None:
            offset = raw
            smooth = None

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)