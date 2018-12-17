# https://github.com/ageitgey/face_recognition
from face_recognition import face_locations


class GetFaceDL:

    def __init__(self):
        return

    def __call__(self, gray):
        faces = face_locations(gray)
        # Convert the coordinates
        return [[xl, yt, xr-xl, yb-yt] for yt, xr, yb, xl in faces]
