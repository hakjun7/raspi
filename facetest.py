import numpy as np
import cv2
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier(
    'opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
image = cv2.imread('./seokcard.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12,8))
plt.imshow(grayImage, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

faces = face_cascade.detectMultiScale(grayImage, 1.03, 5)

print(faces.shape)
print("Number of faces detected: " + str(faces.shape[0]))