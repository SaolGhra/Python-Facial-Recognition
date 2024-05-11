# face_detection.py

import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
    )

    face_coordinates = []
    for x, y, w, h in faces:
        face_coordinates.append((x, y, x + w, y + h))  # (left, top, right, bottom)

    return face_coordinates
