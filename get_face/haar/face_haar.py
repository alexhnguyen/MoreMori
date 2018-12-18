import cv2
import os

class GetFaceHaar():

    def __init__(self):
        base = os.path.dirname(os.path.realpath(__file__))
        self.face_cascade = cv2.CascadeClassifier(os.path.join(base, 'opencv_haar/haarcascade_frontalface_default.xml'))
        return

    def __call__(self, gray):
        return self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
