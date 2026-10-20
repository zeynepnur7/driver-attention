# stage1/04_video_file.py
import time
import cv2

cap = cv2.VideoCapture("data/bunny.mov")

if not cap.isOpened():
    raise RuntimeError("Video açılamadı")

video_fps = cap.get(cv2.CAP_PROP_FPS)
print("Videonun FPS değeri:", video_fps)

prev_time = time.perf_counter()

while True:
    ret, frame = cap.read()
    if not ret:          # video bitti
        break

    now = time.perf_counter()
    fps = 1.0 / (now - prev_time)
    prev_time = now

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(int(1000 / video_fps)) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()