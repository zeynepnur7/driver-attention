import cv2 as cv

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))

cap = cv.VideoCapture(0)
cv.namedWindow("Mouse")
cv.setMouseCallback("Mouse", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for p in points:
        cv.circle(frame, p, 8, (0, 0, 255), -1)   # -1 = içi dolu

    cv.imshow("Mouse", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        points.clear()          # c ile temizle

cap.release()
cv.destroyAllWindows()