# main.py

import cv2
from functions import (
    face_detection,
    facial_landmarks,
    hand_detection,
    hand_landmark_detection,
)


def main():
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Create an instance of HandLandmarkDetector
    hand_landmark_detector = hand_landmark_detection.HandLandmarkDetector()

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = face_detection.detect_faces(frame)

        # Plot facial landmarks on each face
        for face in faces:
            landmarks = facial_landmarks.detect_landmarks(frame, face)
            for x, y in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Detect hands
        hands = hand_detection.detect_hands(frame)

        # Plot hand landmarks on each hand
        for hand in hands:
            frame, hand_landmarks = hand_landmark_detector.detect_landmarks(frame)
            for landmarks in hand_landmarks:
                for x, y in landmarks:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # Display the frame
        cv2.imshow("Facial Landmarks and Hand Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
