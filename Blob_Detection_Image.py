import cv2
import numpy as np
img = cv2.imread("wand sample 2.jpg", 0)
img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
params = cv2.SimpleBlobDetector_Params()

# setting the thresholds
params.minThreshold = 150
params.maxThreshold = 250

# filter by color
params.filterByColor = 1
params.blobColor = 255

# filter by circularity
params.filterByCircularity = 1
params.minCircularity = 0.5

# filter by area
params.filterByArea = 1
params.minArea = 110
params.maxArea = 300


detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(img)
img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("photo", img)
cv2.imshow("photo with blobs", img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
