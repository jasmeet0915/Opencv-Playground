# For camera module
from picamera import PiCamera
from picamera.array import PiRGBArray

import cv2
import subprocess
import time
import pickle
import numpy as np

# initializing Picamera
camera = PiCamera()
camera.framerate = 33
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size = (640, 480))

embedder_path = "/home/pi/Desktop/face_recognition/openface_nn4.small2.v1.t7"
encoder_path = "/home/pi/Desktop/face_recognition/encoder.pickle"
classifier_path = "/home/pi/Desktop/face_recognition/classifier.pickle"

classifier = pickle.loads(open(classifier_path, "rb").read())
le = pickle.loads(open(encoder_path, "rb").read())
embedder = cv2.dnn.readNetFromTorch(embedder_path)


time.sleep(0.1)
face_cascade = cv2.CascadeClassifier("/home/pi/Desktop/face_recognition/haarcascade_frontalface_default.xml")
for image in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    frame = image.array
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

    if len(faces) > 0:
        # cv2.imwrite("/home/pi/Desktop/face.jpg", frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (96, 96))
            face_blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(face_blob)
            vec = embedder.forward()
            # prediction\
            preds = classifier.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            text = "{}: {:.2f}%".format(name, proba * 100)
            Y = y-10
            cv2.putText(frame, text, (x, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("video", frame)
    rawCapture.truncate(0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cv2.destroyAllWindows()
