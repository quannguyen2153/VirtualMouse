import cv2
import time
import mediapipe as mp
import numpy as np
from enum import Enum

class HandType(Enum):
    LEFT_HAND = "Left"
    RIGHT_HAND = "Right"

class HandDetector():
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.static_image_mode,
                                        max_num_hands=self.max_num_hands,
                                        model_complexity=self.model_complexity,
                                        min_detection_confidence=self.min_detection_confidence,
                                        min_tracking_confidence=self.min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

        self.tip_indices = [4, 8, 12, 16, 20]

    def process(self, image):
        self.processing_image = image

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)

        self.hand_landmarks = {
            HandType.LEFT_HAND.value: [],
            HandType.RIGHT_HAND.value: []
        }
        self.scaled_hand_landmarks = {
            HandType.LEFT_HAND.value: [],
            HandType.RIGHT_HAND.value: []
        }
        self.hands_detected = []

        if self.results.multi_hand_landmarks:
            for hand_landmarks, hand_type in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                hand_label = hand_type.classification[0].label

                self.hands_detected.append(hand_label)

                landmarks_position = []
                scaled_landmarks_position = []

                for landmark in hand_landmarks.landmark:
                    sx = int(landmark.x * self.processing_image.shape[1])
                    sy = int(landmark.y * self.processing_image.shape[0])
                    scaled_landmarks_position.append([sx, sy, landmark.z])
                    
                    landmarks_position.append([landmark.x, landmark.y, landmark.z])

                self.hand_landmarks[hand_label] = landmarks_position
                self.scaled_hand_landmarks[hand_label] = scaled_landmarks_position

    def drawHandsConnection(self, hand_box_offset=10, hand_label_offset=10, flipped=False):
        img = self.processing_image

        def flipHandLabel(hand_label):
            if hand_label == HandType.LEFT_HAND.value:
                return HandType.RIGHT_HAND.value
            return HandType.LEFT_HAND.value

        if self.results.multi_hand_landmarks:
            for hand_landmarks, hand_type in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                hand_label = hand_type.classification[0].label
                scaled_landmarks_position = np.array(self.getLandmarksPosition(hand_label, True))

                min_values = np.min(scaled_landmarks_position, axis=0)
                max_values = np.max(scaled_landmarks_position, axis=0)

                if flipped:
                    hand_label = flipHandLabel(hand_label)

                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(img, (int(min_values[0]) - hand_box_offset, int(min_values[1]) - hand_box_offset), (int(max_values[0]) + hand_box_offset, int(max_values[1]) + hand_box_offset), (0, 0, 255), 2)
                cv2.putText(img, hand_label, (int(min_values[0]) - hand_box_offset, int(min_values[1]) - hand_box_offset - hand_label_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img

    def getHandsDetected(self):
        return self.hands_detected

    def getLandmarksPosition(self, hand_label, scale=False):
        return self.scaled_hand_landmarks[hand_label] if scale else self.hand_landmarks[hand_label]

    def getRaisedFingers(self, flipped=False):
        raised_fingers = {
            HandType.LEFT_HAND.value: [],
            HandType.RIGHT_HAND.value: []
        }

        if not self.results.multi_hand_landmarks:
            return raised_fingers

        hand_labels = tuple(self.hand_landmarks.keys())

        def checkThumb(hand_label, flipped):
            nonlocal raised_fingers

            if flipped:
                if self.hand_landmarks[hand_label][1][0] > self.hand_landmarks[hand_label][17][0]:
                    raised_fingers[hand_label].append(self.hand_landmarks[hand_label][self.tip_indices[0]][0] > self.hand_landmarks[hand_label][self.tip_indices[0] - 1][0] and self.hand_landmarks[hand_label][0][1] < self.hand_landmarks[hand_label][9][1])
                else:
                    raised_fingers[hand_label].append(self.hand_landmarks[hand_label][self.tip_indices[0]][0] < self.hand_landmarks[hand_label][self.tip_indices[0] - 1][0] and self.hand_landmarks[hand_label][0][1] < self.hand_landmarks[hand_label][9][1])
            else:
                if self.hand_landmarks[hand_label][1][0] > self.hand_landmarks[hand_label][17][0]:
                    raised_fingers[hand_label].append(self.hand_landmarks[hand_label][self.tip_indices[0]][0] > self.hand_landmarks[hand_label][self.tip_indices[0] - 1][0] and self.hand_landmarks[hand_label][0][1] > self.hand_landmarks[hand_label][9][1])
                else:
                    raised_fingers[hand_label].append(self.hand_landmarks[hand_label][self.tip_indices[0]][0] < self.hand_landmarks[hand_label][self.tip_indices[0] - 1][0] and self.hand_landmarks[hand_label][0][1] > self.hand_landmarks[hand_label][9][1])

        if len(self.hand_landmarks[HandType.LEFT_HAND.value]) > 0 and len(self.hand_landmarks[HandType.RIGHT_HAND.value]) > 0:
            # Thumb
            checkThumb(HandType.LEFT_HAND.value, flipped)
            checkThumb(HandType.RIGHT_HAND.value, flipped)

            # Others
            if flipped:
                for idx in range(1, 5):
                    raised_fingers[HandType.LEFT_HAND.value].append(self.hand_landmarks[HandType.LEFT_HAND.value][self.tip_indices[idx]][1] > self.hand_landmarks[HandType.LEFT_HAND.value][self.tip_indices[idx] - 2][1] and self.hand_landmarks[HandType.LEFT_HAND.value][0][1] < self.hand_landmarks[HandType.LEFT_HAND.value][9][1])
                    raised_fingers[HandType.RIGHT_HAND.value].append(self.hand_landmarks[HandType.RIGHT_HAND.value][self.tip_indices[idx]][1] > self.hand_landmarks[HandType.RIGHT_HAND.value][self.tip_indices[idx] - 2][1] and self.hand_landmarks[HandType.RIGHT_HAND.value][0][1] < self.hand_landmarks[HandType.RIGHT_HAND.value][9][1])
            else:
                for idx in range(1, 5):
                    raised_fingers[HandType.LEFT_HAND.value].append(self.hand_landmarks[HandType.LEFT_HAND.value][self.tip_indices[idx]][1] < self.hand_landmarks[HandType.LEFT_HAND.value][self.tip_indices[idx] - 2][1] and self.hand_landmarks[HandType.LEFT_HAND.value][0][1] > self.hand_landmarks[HandType.LEFT_HAND.value][9][1])
                    raised_fingers[HandType.RIGHT_HAND.value].append(self.hand_landmarks[HandType.RIGHT_HAND.value][self.tip_indices[idx]][1] < self.hand_landmarks[HandType.RIGHT_HAND.value][self.tip_indices[idx] - 2][1] and self.hand_landmarks[HandType.RIGHT_HAND.value][0][1] > self.hand_landmarks[HandType.RIGHT_HAND.value][9][1])

        elif len(self.hand_landmarks[HandType.LEFT_HAND.value]) > 0:
            # Thumb
            checkThumb(HandType.LEFT_HAND.value, flipped)

            # Others
            if flipped:
                for idx in range(1, 5):
                    raised_fingers[HandType.LEFT_HAND.value].append(self.hand_landmarks[HandType.LEFT_HAND.value][self.tip_indices[idx]][1] > self.hand_landmarks[HandType.LEFT_HAND.value][self.tip_indices[idx] - 2][1] and self.hand_landmarks[HandType.LEFT_HAND.value][0][1] < self.hand_landmarks[HandType.LEFT_HAND.value][9][1])
            else:
                for idx in range(1, 5):
                    raised_fingers[HandType.LEFT_HAND.value].append(self.hand_landmarks[HandType.LEFT_HAND.value][self.tip_indices[idx]][1] < self.hand_landmarks[HandType.LEFT_HAND.value][self.tip_indices[idx] - 2][1] and self.hand_landmarks[HandType.LEFT_HAND.value][0][1] > self.hand_landmarks[HandType.LEFT_HAND.value][9][1])

        else:
            # Thumb
            checkThumb(HandType.RIGHT_HAND.value, flipped)
            
            # Others
            if flipped:
                for idx in range(1, 5):
                    raised_fingers[HandType.RIGHT_HAND.value].append(self.hand_landmarks[HandType.RIGHT_HAND.value][self.tip_indices[idx]][1] > self.hand_landmarks[HandType.RIGHT_HAND.value][self.tip_indices[idx] - 2][1] and self.hand_landmarks[HandType.RIGHT_HAND.value][0][1] < self.hand_landmarks[HandType.RIGHT_HAND.value][9][1])
            else:
                for idx in range(1, 5):
                    raised_fingers[HandType.RIGHT_HAND.value].append(self.hand_landmarks[HandType.RIGHT_HAND.value][self.tip_indices[idx]][1] < self.hand_landmarks[HandType.RIGHT_HAND.value][self.tip_indices[idx] - 2][1] and self.hand_landmarks[HandType.RIGHT_HAND.value][0][1] > self.hand_landmarks[HandType.RIGHT_HAND.value][9][1])

        return raised_fingers

    def get2DDistance(self, hand_label, landmark_index1, landmark_index2, scale=False):
        if len(self.hand_landmarks[hand_label]) <= 0:
            return None
        if scale:
            return np.linalg.norm(np.array(self.scaled_hand_landmarks[hand_label][landmark_index1][:-1]) - np.array(self.scaled_hand_landmarks[hand_label][landmark_index2][:-1]))
        return np.linalg.norm(np.array(self.hand_landmarks[hand_label][landmark_index1][:-1]) - np.array(self.hand_landmarks[hand_label][landmark_index2][:-1]))

    def isTouching(self, hand_label, finger_index1, finger_index2, sensitivity=0.005):
        return self.get2DDistance(hand_label, self.tip_indices[finger_index1], self.tip_indices[finger_index2], scale=False) < -0.2747216 * ((self.hand_landmarks[hand_label][self.tip_indices[finger_index1]][2] + self.hand_landmarks[hand_label][self.tip_indices[finger_index2]][2]) / 2) + 0.00974701 + sensitivity


def getFPS():
    if not hasattr(getFPS, "previous_time"):
        getFPS.previous_time = 0
    if not hasattr(getFPS, "current_time"):
        getFPS.current_time = 0
    
    getFPS.current_time = time.time()
    fps = 1 / (getFPS.current_time - getFPS.previous_time)
    getFPS.previous_time = getFPS.current_time

    return fps