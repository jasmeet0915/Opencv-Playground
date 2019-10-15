from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

embeddings_path = "/home/jasmeet/PycharmProjects/opencv_playground/opencv_and_dlib/face_recognition/embeddings.pickle"
classifier_path = "/home/jasmeet/PycharmProjects/opencv_playground/opencv_and_dlib/face_recognition/classifier.pickle"
encoder_path = "/home/jasmeet/PycharmProjects/opencv_playground/opencv_and_dlib/face_recognition/encoder.pickle"
print("loading extracted embedding features file ")
data = pickle.loads(open(embeddings_path, "rb").read())
print(data)

print("encoding labels")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("training model")
classifier = SVC(C=1.0, kernel="linear", probability=True)
classifier.fit(data["embeddings"], labels)
print("done training")

print("saving classifier")
f = open(classifier_path, "wb")
f.write(pickle.dumps(classifier, protocol=2))
f.close()

print("saving encoder file")
f = open(encoder_path, "wb")
f.write(pickle.dumps(le, protocol=2))
f.close()