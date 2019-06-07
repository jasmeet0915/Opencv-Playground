import cv2
import numpy as np
points = []
cap = cv2.VideoCapture("wand W video.mp4")

params = cv2.SimpleBlobDetector_Params()

# setting the thresholds
params.minThreshold = 150
params.maxThreshold = 250

# filter by color
params.filterByColor = 1
params.blobColor = 255

# filter by circularity
params.filterByCircularity = 1
params.minCircularity = 0.68

# filter by area
params.filterByArea = 1
params.minArea = 400
# params.maxArea = 1500

detector = cv2.SimpleBlobDetector_create(params)
maxlen = 90

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    keypoints = detector.detect(frame)
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if len(keypoints) != 0:
        if len(points) > maxlen:
            points = []
        points_array = cv2.KeyPoint_convert(keypoints)
        points.append(points_array[0])
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame_with_keypoints, tuple(points[i-1]), tuple(points[i]), (0, 255, 0), 15)
    lower_green = np.array([0, 255, 0])
    upper_green = np.array([0, 255, 0])
    frame_with_keypoints = cv2.inRange(frame_with_keypoints, lower_green, upper_green)
    cv2.imshow("real video", frame)
    cv2.imshow("video with blobs", frame_with_keypoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()