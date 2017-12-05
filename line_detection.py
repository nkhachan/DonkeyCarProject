import cv2
import numpy as np

def white_mask(img):
	""" Apply white mask to color image. 
    	parameters: img Image to process
    	returns : white_image Image with only white not blacked out 
	"""
	
	#Range of mask 
	lower_white = np.array([0, 0, 0], dtype = "uint8")
	upper_white = np.array([255, 255, 255], dtype="uint8")
	mask_white = cv2.inRange(img, 200, 255)
	return mask_white

def yellow_mask(img):

	#Range of yellow 
	lower_yellow = np.array([50, 50, 50], dtype = "uint8")
	upper_yellow = np.array([110, 255, 255], dtype="uint8")
	mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)
	return mask_yellow

def find_lines(img):

	gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gaussian_fix = cv2.GaussianBlur(gray_image,(3,3),0)
	return cv2.Canny(gaussian_fix, 50,160)

def process_image(img):

	#HSV Color Space
	lines = find_lines(img)
	img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	yellow = yellow_mask(img_hsv)
	white = white_mask(img_hsv)

	arr = np.zeros((120,160,3))
	arr[:,:,1] = yellow
	print type(arr)
	print arr.shape
	
	#Display Image
	cv2.imshow('image',img_hsv)
	cv2.imshow('image',yellow)
	cv2.imshow('image',arr)
#	cv2.imshow('image',white)
	cv2.waitKey()



img = cv2.imread('../d2/data/tub_3_17-10-23/6_cam-image_array_.jpg')
process_image(img)


