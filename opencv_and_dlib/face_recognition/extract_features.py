import cv2
import dlib
import os
import pickle


def shape_to_bb(shape):
    x = shape.left()
    y = shape.top()
    w = shape.right() - x
    h = shape.bottom() - y
    return x, y, w, h


knownEmbeddings = []
knownNames = []
detector = dlib.get_frontal_face_detector()

print("...Loading embedding model...")
embedder = cv2.dnn.readNetFromTorch("/home/jasmeet/PycharmProjects/opencv_playground/opencv_and_dlib/face_recognition/openface_nn4.small2.v1.t7")

root = "/home/jasmeet/PycharmProjects/opencv_playground/opencv_and_dlib/face_recognition/dataset"
embeddings = "/home/jasmeet/PycharmProjects/opencv_playground/opencv_and_dlib/face_recognition/embeddings.pickle"
i=0

for (roots, dirs, files) in os.walk(root, topdown=True):
    for file in files:
        imgpath = str(roots)+str("/")+str(file)
        img = cv2.imread(str(imgpath))
        img = cv2.resize(img, (300, 300))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(img_gray, 1)
        for rect in rects:
            (x, y, w, h) = shape_to_bb(rect)
            face_roi = img[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (96, 96))
            face_blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            print("...Creating embeddings...")
            embedder.setInput(face_blob)
            vec = embedder.forward()
            print(os.path.basename(roots))
            knownEmbeddings.append(vec.flatten())
            knownNames.append(os.path.basename(roots))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(embeddings, "wb")
f.write(pickle.dumps(data))
f.close()
