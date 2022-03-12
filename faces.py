import os
import cv2
from PIL import Image
import numpy as np
import pickle

base = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base, "images")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# LBPH is a data algorithm that is used mostly for face recignition, there is a lot other than that, like AdaBoost algorithm etc
recognizer = cv2.face.LBPHFaceRecognizer_create()

currentid = 0
label_ids = {}

x_train = []
y_labels = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            print(label, path)

            if not label in label_ids:
                label_ids[label] = currentid
                currentid += 1
            id_ = label_ids[label]
            print(label_ids)

            pil_image = Image.open(path).convert(
                "L")  # converting into grayscale
            image_array = np.array(pil_image, "uint8")
            print(image_array)

            whatface = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in whatface:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)

# the with open as model is a model to open a new file called "labels.pickle" and assigned it to "f" and "w" means the model (what are you trying to do with it)
# in this case "wb" means "writing binary" that means we want to write in the "f" file in binary mode.
with open("labels.pickle", "wb") as f:
    # pickle.dump means we will dump up the labels_id which is full of id that we made before into the file "f" that was created before
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
