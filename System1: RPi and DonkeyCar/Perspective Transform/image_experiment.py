import cv2
import os
import glob
import numpy as np

directory_link = '/Users/aamirrasheed/Documents/UCSD/Schoolwork/Senior/FQ17/MAE 198/working/data/tub_16_half'
filename = os.path.join(directory_link, '5496_cam-image_array_.jpg')
image = cv2.imread(filename, 0)
_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
print("done")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)


def processImages(input_files_pattern, output_directory, processor):
    image_paths = glob.glob(input_files_pattern)

    for image_path in image_paths:
        image = cv2.imread(filename, 0)
        output_image = processor(image)
        cv2.imwrite(output_directory + image_path, output_image)

def findCorners(image, corner_dims):
    ret, corners = cv2.findChessboardCorners(image, corner_dims)

    imgpoints = []
    objpoints = []

    objpoints.append(objp)
    cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners)

    cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
    cv2.imshow('img', img)
    cv2.waitKey(500)
