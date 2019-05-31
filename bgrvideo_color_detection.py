import cv2
import time
cap = cv2.VideoCapture(2)
'''count = 0
sum_x = 0
avg_x = 0
sum_y = 0
avg_y = 0'''
threshold = 180


def color_dist(b1, g1, r1):
    dist = (int(b1) * int(b1)) + (int(g1) * int(g1)) + (int(255-r1) * int(255-r1))
    return dist


while True:
    _, frame = cap.read()
    start = time.time()
    frame2 = cv2.resize(frame, (frame.shape[1]//5, frame.shape[0]//5))

    for y in range(frame2.shape[0]):
        for x in range(frame2.shape[1]):
            b_value = frame2[y, x][0]
            g_value = frame2[y, x][1]
            r_value = frame2[y, x][2]
            dist_color = color_dist(b_value, g_value, r_value)
            if dist_color <= threshold * threshold:
                frame[y*5, x*5] = [0, 0, 0]
                frame2[y, x] = [0, 0, 0]
    end = time.time()
    cv2.imshow("final video", frame2)
    cv2.imshow("real video", frame)
    print(end-start)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


