import cv2
import numpy as np
import dlib

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def shape_to_numpy(shape):
    # initializing a numpy array with 68 rows 2 columns with all elements as 0 to hold coordinates from shape detector
    points = np.zeros(shape=(68, 2), dtype='int')
    for i in range(0, 68):
        points[i] = (shape.part(i).x, shape.part(i).y)
    return points


facial_landmarks = {
    "mouth": (48, 68),
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "nose": (27, 35),
    "jawline": (0, 17)
}


detector = dlib.get_frontal_face_detector()
# or read faces.jpeg image
img = cv2.imread("akshay_kumar.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


rects = detector(gray_img, 2)

for rect in rects:
    shape = predictor(gray_img, rect)
    shape = shape_to_numpy(shape)
    (x, y, w, h) = rect_to_bb(rect)
    #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
    for name in facial_landmarks:
        (i, j) = facial_landmarks[name]
        if name == "jawline":
            for (c1, c2) in shape[i:j]:
                cv2.circle(img, (c1, c2), 1, (0, 0, 255), -1)
            cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("img", img)
    roi = img[y:y+h, x:x+w]
    cv2.imshow("roi", roi)


cv2.waitKey(0)
cv2.destroyAllWindows()