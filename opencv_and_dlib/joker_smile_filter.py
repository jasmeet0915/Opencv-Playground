import cv2
import dlib
import numpy as np
import math
from imutils import rotate_bound


def shape_to_numpy(shape):
    points = np.zeros((68, 2), 'int')
    for i in range(0, 68):
        points[i] = (shape.part(i).x, shape.part(i).y)

    return points


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def calculate_inclination(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point1[1]
    slope_angle = 180/math.pi * float(y2-y1/x2-x1)
    return slope_angle


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img = cv2.imread("akshay_kumar.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

smile = cv2.imread("joker_smile.png", -1)

# extracting alpha channel from the image as a separate image for alpha mask fo foreground
alpha_mask = smile[:, :, 3]
# converting the smile image back to BGR with only 3 channel without alpha channel
smile = smile[:, :, 0:3]

alpha_mask_inv = cv2.bitwise_not(alpha_mask)

rects = detector(img_gray, 2)

for rect in rects:
    shape = predictor(img_gray, rect)
    shape = shape_to_numpy(shape)
    (x, y, w, h) = rect_to_bb(rect)
    (x1, y1) = shape[49]
    (x2, y2) = shape[55]
    smile_width = 2*(x2 - x1)
    smile_height = smile_width * smile.shape[0]//smile.shape[1]
    smile = cv2.resize(smile, (smile_width, smile_height), interpolation=cv2.INTER_AREA)
    alpha_mask = cv2.resize(alpha_mask, (smile_width, smile_height), interpolation=cv2.INTER_AREA)
    alpha_mask_inv = cv2.resize(alpha_mask_inv, (smile_width, smile_height), interpolation=cv2.INTER_AREA)
    roi = img[y1:y1 + smile_height, x1:x1 + smile_width] #smile roi
    cv2.imshow("roi", roi)
    roi_bg = cv2.bitwise_and(roi, roi, mask=alpha_mask_inv)
    roi_fg = cv2.bitwise_and(smile, smile, mask=alpha_mask)
    final = cv2.add(roi_bg, roi_fg)
    cv2.imshow("final", final)
    img[y1:y1+smile_height, x1:x1+smile_width] = final


cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()