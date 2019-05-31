import cv2
import time
cap = cv2.VideoCapture(2)
count = 0
sum_x = 0
avg_x = 0
sum_y = 0
avg_y = 0
#threshold = 200


'''def color_dist(b1, g1, r1):
    dist = (int(b1) * int(b1)) + (int(g1) * int(g1)) + (int(255-r1) * int(255-r1))
    return dist'''


while True:
    _, frame = cap.read()
    start = time.time()
    frame2 = cv2.resize(frame, (frame.shape[1]//20, frame.shape[0]//20))
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)


    for y in range(frame2.shape[0]):
        for x in range(frame2.shape[1]):
            h_value = frame2[y, x][0]
            s_value = frame2[y, x][1]
            v_value = frame2[y, x][2]
            if h_value in range(21) or h_value in range(160, 181):
                if s_value > 100 and v_value > 60:
                    sum_x = sum_x + x
                    sum_y = sum_y + y
                    count = count + 1

    avg_x = sum_x//count
    avg_y = sum_y//count
    cv2.circle(frame, (avg_x, avg_y), 15, (255, 255, 255), 4)
    end = time.time()
    cv2.imshow("processing video", frame2)
    cv2.imshow("final video", frame)
    print(end-start)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
