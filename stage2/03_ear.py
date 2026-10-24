import math
import cv2 as cv
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.21      # kendi yüzüne göre ayarlayacaksın
MIN_CLOSED_FRAMES = 2     # kaç ardışık karede kapalıysa kırpma sayılsın


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_points(landmarks, indices, w, h):
    # piksel koordinatına çevir (normalize değerle oran hesaplamak hatalı olur)
    return [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]


def ear(p):
    # p = [p1, p2, p3, p4, p5, p6]
    vertical = dist(p[1], p[5]) + dist(p[2], p[4])
    horizontal = 2.0 * dist(p[0], p[3])
    return vertical / horizontal


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı")

timestamp_ms = 0
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

        timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            lms = result.face_landmarks[0]
            right = eye_points(lms, RIGHT_EYE, w, h)
            left = eye_points(lms, LEFT_EYE, w, h)

            avg_ear = (ear(right) + ear(left)) / 2.0

            # sadece göz noktalarını çiz
            for x, y in right + left:
                cv.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            # göz kırpma sayımı
            if avg_ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                if closed_frames >= MIN_CLOSED_FRAMES:
                    blink_count += 1
                closed_frames = 0

            status = "KAPALI" if avg_ear < EAR_THRESHOLD else "ACIK"
            color = (0, 0, 255) if status == "KAPALI" else (0, 255, 0)

            cv.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv.putText(frame, f"Goz: {status}", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv.putText(frame, f"Kirpma: {blink_count}", (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv.putText(frame, "Yuz bulunamadi", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv.imshow("EAR", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)