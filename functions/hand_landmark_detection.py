# hand_landmark_detection.py

import cv2
import mediapipe as mp


class HandLandmarkDetector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def detect_landmarks(self, frame, draw=True):
        print("detect_landmarks method called")
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hand landmarks
        results = self.hands.process(rgb_frame)

        # Initialize an empty list to store detected landmarks
        landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    # Convert normalized landmarks to pixel coordinates
                    h, w, c = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))
                    if draw:
                        # Draw the landmarks on the frame
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                landmarks_list.append(landmarks)

        return frame, landmarks_list
