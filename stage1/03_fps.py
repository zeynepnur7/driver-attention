import cv2 as cv
import time


cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera can not be opened")

prev_time = time.perf_counter() #save the inital time

while True:
    ret, frame = cap.read() #take continous frame from camera
    if not ret:
        print("Frame can't read")
        break

    now = time.perf_counter() #save the present time
    fps = 1.0 / (now - prev_time)
    prev_time = now

    cv.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),                    # sol üst köşe
        cv.FONT_HERSHEY_SIMPLEX,
        1,                           # yazı boyutu
        (0, 255, 0),                 # yeşil (BGR)
        2,                           # kalınlık
    )
    
    cv.imshow("Webcam + FPS", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)