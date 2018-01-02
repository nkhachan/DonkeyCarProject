import cv2
from carcode import *

img_path = 'pic.jpg'
img = colorProc2(img_path)
og = cv2.imread(img_path)

cv2.imshow('adf',img)
cv2.imshow('og',og)
cv2.waitKey(0)
