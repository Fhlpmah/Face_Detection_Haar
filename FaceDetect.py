import cv2
import numpy as np
import time

face_cascade = cv2.CascadeClassifier(
    "haar-cascade/data/haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture("videos/video3.mp4")

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)

    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_basic = frame[y:y+h, x:x+w]
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (0, 0, 0)  # BGR
        stroke = 2
        end_cordx = x + w
        end_cordy = y + h
        cv2.rectangle(frame, (x, y), (end_cordx, end_cordy), color, stroke)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
