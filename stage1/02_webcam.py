import cv2 as cv

cap = cv.VideoCapture(0)    # 0 = default camera

if not cap.isOpened():
    raise RuntimeError("Camera can not opened")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare okunamadı")
        break
    frame = cv.flip(frame, 1)   # mirror
    cv.imshow("Webcam", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)   