import cv2
import numpy as np
import dlib


def shape_to_numpy(shape):
    points = np.zeros((68, 2), 'int')
    for i in range(0, 68):
        points[i] = (shape.part(i).x, shape.part(i).y)

    return points


lower_green = np.array([0, 255, 0])
upper_green = np.array([0, 255, 0])

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    #frame = cv2.flip(frame, 0)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(frame_gray, 1)
    for rect in rects:
        shape = predictor(frame_gray, rect)
        shape = shape_to_numpy(shape)
        #cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    frame_with_landmarks = cv2.inRange(frame, lower_green, upper_green)
    cv2.imshow("video2", frame_with_landmarks)
    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()



