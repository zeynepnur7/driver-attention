import cv2 as cv
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

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
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)          # MediaPipe RGB ister
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms += 33                                   # yaklaşık 30 FPS
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            h, w = frame.shape[:2]
            for lm in result.face_landmarks[0]:
                x, y = int(lm.x * w), int(lm.y * h)          # normalize -> piksel
                cv.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv.imshow("Face Landmarks", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)