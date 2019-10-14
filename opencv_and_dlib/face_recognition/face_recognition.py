import cv2
import dlib


def shape_to_bb(shape):
    x = shape.left()
    y = shape.top()
    w = shape.right() - x
    h = shape.bottom() - y
    return x, y, w, h


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(frame_gray, 1)
    if len(rects) > 0:
        for rect in rects:
            (x, y, w, h) = shape_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)
    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()