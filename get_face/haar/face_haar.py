import cv2
import os

class GetFaceHaar():

    def __init__(self):
        self.face_cascade = self.setup_cascade()['face_cascade']
        return

    def __call__(self, gray):
        return self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    @staticmethod
    def setup_cascade():
        base = os.path.dirname(os.path.realpath(__file__))
        face_cascade = cv2.CascadeClassifier(os.path.join(base, 'opencv_haar/haarcascade_frontalface_default.xml'))
        eye_cascade = cv2.CascadeClassifier(os.path.join(base, 'opencv_haar/haarcascade_eye.xml'))
        return {'face_cascade': face_cascade,
                'eye_cascade': eye_cascade}
