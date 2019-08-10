import cv2
import numpy
import dlib


img = cv2.imread("faces.jpeg")
#img = cv2.resize(img, (img.shape[1]//10, img.shape[0]//2), interpolation=cv2.INTER_AREA)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread("faces.jpeg")
#img2 = cv2.resize(img2, (img2.shape[1]//10, img2.shape[0]//2), interpolation=cv2.INTER_AREA)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
detector = dlib.get_frontal_face_detector()
rects = detector(gray_img2, 2)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

for rect in rects:
    cv2.rectangle(img2, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 255), 2)


cv2.imshow("faces_haar", img)
cv2.imshow("faces_dlib", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


