import numpy as np
import cv2


def colorProc2(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) #change to correct input color space

    # limits of yellow mask
    lower_yellow = np.array([70, 120, 40], dtype='uint8')
    upper_yellow = np.array([90, 170, 70], dtype='uint8')
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # limits of white mask
    lower_white = np.array([0, 200, 0], dtype='uint8')
    upper_white = np.array([255, 255, 255], dtype='uint8')
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    #y = cv2.bitwise_and(img,img,mask=mask_yellow)
    y = cv2.cvtColor(mask_yellow,cv2.COLOR_GRAY2BGR)
    y[:,:,0] = mask_yellow*255

    w = cv2.bitwise_and(img,img,mask=mask_white)


    # reduce noise
    gauss_gray = cv2.GaussianBlur(w, (3, 3), 0)

    # canny edge
    low_threshold = 50
    high_threshold = 160
    w = cv2.Canny(gauss_gray, low_threshold, high_threshold)
    w = cv2.cvtColor(w,cv2.COLOR_GRAY2BGR)

    yw = y + w

    img = yw
    vlim1 = 30
    vlim2 = 105
    hlim1 = 10
    hlim2 = 150

    h, w, d = img.shape
    mask = np.ones([h, w, d])
    mask[0:vlim1, :, :] = np.zeros([vlim1, w, d])
    mask[vlim2:h, :, :] = np.zeros([(h - vlim2), w, d])
    mask[:, 0:hlim1, :] = np.zeros([h, hlim1, d])
    mask[:, hlim2:w, :] = np.zeros([h, w - hlim2, d])

    mask = np.zeros(img.shape, dtype="uint8")
    img = img[vlim1:vlim2, hlim1:hlim2, :]
    mask[vlim1:vlim2, hlim1:hlim2, :] = cv2.bitwise_or(mask[vlim1:vlim2, hlim1:hlim2, :], img)
    img = mask

    return img

def colorProc(img_path):

    """ Function to process image 
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)      #CHANGE THE COLOR SPACE ON THIS

    # limits of yellow mask
    lower_yellow = np.array([50, 50, 50], dtype='uint8')
    upper_yellow = np.array([110, 255, 255], dtype='uint8')
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # limits of white mask
    lower_white = np.array([0, 0, 170], dtype='uint8')
    upper_white = np.array([255, 85, 255], dtype='uint8')
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    #y = cv2.bitwise_and(img,img,mask=mask_yellow)
    y = cv2.cvtColor(mask_yellow,cv2.COLOR_GRAY2BGR)
    y[:,:,0] = mask_yellow*255

    w = cv2.bitwise_and(img,img,mask=mask_white)


    # reduce noise
    gauss_gray = cv2.GaussianBlur(w, (3, 3), 0)

    # canny edge
    low_threshold = 50
    high_threshold = 160
    w = cv2.Canny(gauss_gray, low_threshold, high_threshold)
    w = cv2.cvtColor(w,cv2.COLOR_GRAY2BGR)

    yw = y + w

    img = yw
    vlim1 = 30
    vlim2 = 105
    hlim1 = 10
    hlim2 = 150

    h, w, d = img.shape
    mask = np.ones([h, w, d])
    mask[0:vlim1, :, :] = np.zeros([vlim1, w, d])
    mask[vlim2:h, :, :] = np.zeros([(h - vlim2), w, d])
    mask[:, 0:hlim1, :] = np.zeros([h, hlim1, d])
    mask[:, hlim2:w, :] = np.zeros([h, w - hlim2, d])

    mask = np.zeros(img.shape, dtype="uint8")
    img = img[vlim1:vlim2, hlim1:hlim2, :]
    mask[vlim1:vlim2, hlim1:hlim2, :] = cv2.bitwise_or(mask[vlim1:vlim2, hlim1:hlim2, :], img)
    img = mask

    return img

def lineProc(img_path):
    """ Function to remove top portion of the image that doesn't
        have the road.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # limits of yellow mask
    lower_white = np.array([0, 0, 0], dtype='uint8')
    upper_white = np.array([255, 255, 255], dtype='uint8')
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # limits of yellow mask
    lower_yellow = np.array([50, 50, 50], dtype='uint8')
    upper_yellow = np.array([110, 255, 255], dtype='uint8')
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    y = cv2.bitwise_and(gray, mask_yellow)
    w = cv2.bitwise_and(gray, mask_white)
    yw = y + w

    # reduce noise
    gauss_gray = cv2.GaussianBlur(yw, (3, 3), 0)

    # canny edge
    low_threshold = 50
    high_threshold = 160
    img = cv2.Canny(gauss_gray, low_threshold, high_threshold)

    vlim1 = 30
    vlim2 = 105
    hlim1 = 10
    hlim2 = 150

    h, w = img.shape
    mask = np.ones([h, w])
    mask[0:vlim1, :] = np.zeros([vlim1, w])
    mask[vlim2:h, :] = np.zeros([(h - vlim2), w])
    mask[:, 0:hlim1] = np.zeros([h, hlim1])
    mask[:, hlim2:w] = np.zeros([h, w - hlim2])

    mask = np.zeros(img.shape, dtype="uint8")
    img = img[vlim1:vlim2, hlim1:hlim2]
    mask[vlim1:vlim2, hlim1:hlim2] = cv2.bitwise_or(mask[vlim1:vlim2, hlim1:hlim2], img)
    img = mask

    return img


def carcrop(img_path):
    """ Function to remove top portion of the image that doesn't
        have the road.
    """
    img = cv2.imread(img_path)
    vlim1 = 30
    vlim2 = 105
    hlim1 = 10
    hlim2 = 150

    img = cv2.imread(img_path)
    h, w, d = img.shape
    mask = np.ones([h, w, d])
    mask[0:vlim1, :, :] = np.zeros([vlim1, w, d])
    mask[vlim2:h, :, :] = np.zeros([(h - vlim2), w, d])
    mask[:, 0:hlim1, :] = np.zeros([h, hlim1, d])
    mask[:, hlim2:w, :] = np.zeros([h, w - hlim2, d])

    mask = np.zeros(img.shape, dtype="uint8")
    img = img[vlim1:vlim2, hlim1:hlim2, :]
    mask[vlim1:vlim2, hlim1:hlim2, :] = cv2.bitwise_or(mask[vlim1:vlim2, hlim1:hlim2, :], img)
    img = mask
    return img
