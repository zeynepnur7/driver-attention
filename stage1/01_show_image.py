import cv2 as cv

img = cv.imread('data/cc5.jpeg')

if img is None:
    raise FileNotFoundError("Image can not found, check the path")

print("shape:", img.shape)   # (yükseklik, genişlik, kanal)
print("dtype:", img.dtype)   # uint8

cv.imshow('Monica', img)

cv.waitKey(0)
cv.destroyAllWindows()
