import math
import time

import cv2 as cv
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def rotation_to_euler(R):
    """3x3 dönme matrisinden (pitch, yaw, roll) derece cinsinden."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return tuple(math.degrees(a) for a in (pitch, yaw, roll))


def wrap180(a):
    """Açıyı -180..180 aralığına al."""
    return (a + 180.0) % 360.0 - 180.0


def put(frame, text, y, color=(255, 255, 255)):
    cv.putText(frame, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_facial_transformation_matrixes=True,   # head pose için gerekli
)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı")

t0 = time.perf_counter()
last_ts = -1
offset = (0.0, 0.0, 0.0)    # nötr duruş (c ile ayarlanır)
raw = (0.0, 0.0, 0.0)

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
            M = np.array(mats[0])            # 4x4
            R = M[:3, :3]
            raw = rotation_to_euler(R)

            pitch = wrap180(raw[0] - offset[0])
            yaw = wrap180(raw[1] - offset[1])
            roll = wrap180(raw[2] - offset[2])

            put(frame, f"Pitch: {pitch:6.1f}", 30)
            put(frame, f"Yaw:   {yaw:6.1f}", 60)
            put(frame, f"Roll:  {roll:6.1f}", 90)
            put(frame, f"ham: {raw[0]:.0f} {raw[1]:.0f} {raw[2]:.0f}", 120, (180, 180, 180))
        else:
            put(frame, "Yuz bulunamadi", 30, (0, 0, 255))

        put(frame, "c: notr durus  q: cikis", h - 15, (200, 200, 200))
        cv.imshow("Head Pose", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            offset = raw          # şu anki duruşu sıfır kabul et

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)