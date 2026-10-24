import cv2 as cv
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Sıra: p1 köşe, p2 üst, p3 üst, p4 köşe, p5 alt, p6 alt
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı")

timestamp_ms = 0

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

            for indices, color in ((RIGHT_EYE, (0, 255, 0)), (LEFT_EYE, (255, 0, 0))):
                for n, idx in enumerate(indices, start=1):
                    x = int(lms[idx].x * w)
                    y = int(lms[idx].y * h)
                    cv.circle(frame, (x, y), 3, color, -1)
                    cv.putText(frame, f"p{n}", (x, y - 6),
                               cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv.imshow("Goz noktalari", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)