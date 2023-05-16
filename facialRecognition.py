"""
This is the main code that will be used for facial recognition.

RetinaFace is the main module that will be used for all the facial detection concepts.
Link to the RetinaFace official repository can be found here - https://github.com/serengil/retinaface

OpenCV Python will be used for all image and video processing functionalities in python.

INSERT dataset is used to train the neural network for emotion recognition. 

"""
from deepface import DeepFace
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

print("work")
"""cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.flip(frame, 90)
    cv.imshow('Face', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        #out = cv.imwrite('capture.jpg', Face)
        break

cap.release()
cv.destroyAllWindows()"""

img = cv.imread('happy-girl2.jpeg')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

try:
  predictions= DeepFace.analyze(img)
except:
  print("No face detected")