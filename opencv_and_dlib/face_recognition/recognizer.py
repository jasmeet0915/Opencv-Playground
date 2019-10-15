import dlib
import pickle
import cv2
import numpy as np


def shape_to_bb(shape):
    x = shape.left()
    y = shape.top()
    w = shape.right() - x
    h = shape.bottom() - y
    return x, y, w, h


cap = cv2.VideoCapture(-1)
detector = dlib.get_frontal_face_detector()

#imgpath = "/home/jasmeet/PycharmProjects/opencv_playground/opencv_and_dlib/face_recognition/testing/11.jpg"
embedder_path = "/home/jasmeet/PycharmProjects/opencv_playground/opencv_and_dlib/face_recognition/openface_nn4.small2.v1.t7"
encoder_path = "/home/jasmeet/PycharmProjects/opencv_playground/opencv_and_dlib/face_recognition/encoder.pickle"
classifier_path = "/home/jasmeet/PycharmProjects/opencv_playground/opencv_and_dlib/face_recognition/classifier.pickle"

classifier = pickle.loads(open(classifier_path, "rb").read())
le = pickle.loads(open(encoder_path, "rb").read())

embedder = cv2.dnn.readNetFromTorch(embedder_path)

while True:
    _, img = cap.read()
    print(img.shape)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    img = cv2.flip(img, 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray, 1)
    for rect in rects:
        (x, y, w, h) = shape_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = img[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (96, 96))
        face_blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(face_blob)
        vec = embedder.forward()

        preds = classifier.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]
        text = "{}: {:.2f}%".format(name, proba * 100)
        Y = y-10
        cv2.putText(img, text, (x, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()