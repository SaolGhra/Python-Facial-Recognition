# facial_landmarks.py

import cv2
import dlib

# Load the pre-trained facial landmark detection model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detect_landmarks(frame, face):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (x, y, x2, y2) = face

    rect = dlib.rectangle(x, y, x2, y2)
    landmarks = predictor(gray, rect)

    landmark_points = []
    for i in range(68):
        x_lm = landmarks.part(i).x
        y_lm = landmarks.part(i).y
        landmark_points.append((x_lm, y_lm))

    return landmark_points
