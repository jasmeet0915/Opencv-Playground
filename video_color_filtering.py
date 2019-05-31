import cv2
import numpy as np
import time
cap = cv2.VideoCapture(2)
while True:
    _, frame = cap.read()
    start = time.time()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([75, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    res = cv2.bitwise_and(frame, frame, mask = mask)
    end = time.time()

    cv2.imshow("real video", frame)
    cv2.imshow("hsv video", hsv)
    cv2.imshow("mask video", mask)
    cv2.imshow("final video", res)
    print(end-start)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
