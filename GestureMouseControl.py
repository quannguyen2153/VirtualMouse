import cv2
import numpy as np
import HandTrackingModule as htm
import screeninfo
import mouse

class GestureMouseControl:
    def __init__(self,
                 main_hand,
                 image_width,
                 image_height,
                 monitor_width,
                 monitor_height,
                 click_threshold=0.03,
                 x_frame_reduction=320,
                 y_frame_reduction=180,
                 x_offset=50,
                 y_offset=30,
                 smoothening=5,
                 min_detection_confidence=0.9,
                 min_tracking_confidence=0.9):
        self.main_hand = main_hand
        self.other_hand = htm.HandType.LEFT_HAND.value if self.main_hand == htm.HandType.RIGHT_HAND.value else htm.HandType.RIGHT_HAND.value

        self.image_width = image_width
        self.image_height = image_height
        self.monitor_width = monitor_width
        self.monitor_height = monitor_height
        self.click_threshold = click_threshold
        self.x_frame_reduction = x_frame_reduction
        self.y_frame_reduction = y_frame_reduction
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.smoothening = smoothening
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.hand_detector = htm.HandDetector(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

        self.plocX = 0
        self.plocY = 0
        self.clocX = 0
        self.clocY = 0
        self.is_clicked1 = False
        self.is_clicked2 = False

    def process(self, image):
        self.processing_image = image

        self.hand_detector.process(image)

        self.hands_detected = self.hand_detector.getHandsDetected()

        if self.main_hand in self.hands_detected:
            self.landmarks_position = self.hand_detector.getLandmarksPosition(self.main_hand)
            self.scaled_landmarks_position = self.hand_detector.getLandmarksPosition(self.main_hand, scale=True)

            self.raised_fingers = self.hand_detector.getRaisedFingers()[self.main_hand]
            self.flipped_raised_fingers = self.hand_detector.getRaisedFingers(flipped=True)[self.main_hand]

            if self.other_hand in self.hands_detected:
                self.other_landmarks_position = self.hand_detector.getLandmarksPosition(self.other_hand)
                self.other_scaled_landmarks_position = self.hand_detector.getLandmarksPosition(self.other_hand, scale=True)

                self.other_raised_fingers = self.hand_detector.getRaisedFingers()[self.other_hand]
                self.other_flipped_raised_fingers = self.hand_detector.getRaisedFingers(flipped=True)[self.other_hand]

    def visualize(self, hands_connection=True, bounding_frame=True, flipped=False):
        if hands_connection:
            self.processing_image = self.hand_detector.drawHandsConnection(flipped=flipped)
        if bounding_frame:
            cv2.rectangle(self.processing_image, (self.x_frame_reduction + self.x_offset, self.y_frame_reduction + self.y_offset), (self.image_width - self.x_frame_reduction + self.x_offset, self.image_height - self.y_frame_reduction + self.y_offset), (0, 255, 0), 2)
        
        return self.processing_image

    def is2HandControlActivated(self):
        if not self.is_clicked1 and self.main_hand in self.hands_detected and self.other_hand in self.hands_detected:
            return True
        return False

    def is1HandControlActivated(self):
        if not self.is_clicked2 and self.main_hand in self.hands_detected and self.other_hand not in self.hands_detected:
            return True
        return False

    def extendedControl(self, visualized=False):
        if self.is2HandControlActivated():
            if self.raised_fingers[1] == True and self.raised_fingers[2] == True and self.raised_fingers.count(False) >= 3:
                mouse.release()
                
                if self.hand_detector.isIndexMiddleTouching(self.main_hand):
                    x_mid, y_mid = self.scaled_landmarks_position[12][:-1]

                    if x_mid > self.x_frame_reduction + self.x_offset and x_mid < self.image_width - self.x_frame_reduction + self.x_offset and y_mid > self.y_frame_reduction + self.y_offset and y_mid < self.image_height - self.y_frame_reduction + self.y_offset:
                        
                        if not self.is_clicked2:
                            self.plocX, self.plocY = x_mid, y_mid
                        else:
                            self.x_offset += x_mid - self.plocX
                            self.y_offset += y_mid - self.plocY

                            self.plocX, self.plocY = x_mid, y_mid

                        if visualized:
                            if not self.is_clicked2:
                                cv2.circle(self.processing_image, (x_mid, y_mid), 15, (0, 255, 0), cv2.FILLED)
                            else:
                                cv2.circle(self.processing_image, (x_mid, y_mid), 15, (0, 0, 255), cv2.FILLED)

                        self.is_clicked2 = True

                else:
                    self.is_clicked2 = False
            
            else:
                mouse.release()
                self.is_clicked2 = False

        return self.processing_image

    def controlCursor(self, visualized=False, flipped=True):
        def moveCursor(x, y):
            x_scr = np.interp(x, (self.x_frame_reduction + self.x_offset, self.image_width - self.x_frame_reduction + self.x_offset), (0, self.monitor_width))
            y_scr = np.interp(y, (self.y_frame_reduction + self.y_offset, self.image_height - self.y_frame_reduction + self.y_offset), (0, self.monitor_height))

            self.clocX = self.plocX + (x_scr - self.plocX) / self.smoothening
            self.clocY = self.plocY + (y_scr - self.plocY) / self.smoothening
        
            mouse.move(self.monitor_width - self.clocX if flipped else self.clocX, self.clocY)
            self.plocX, self.plocY = self.clocX, self.clocY

        if self.is1HandControlActivated():
            # Move cursor
            if self.raised_fingers[1] == True and self.raised_fingers.count(False) >= 4:
                mouse.release()

                x_idx, y_idx = self.scaled_landmarks_position[8][:-1]

                moveCursor(x_idx, y_idx)

                self.is_clicked1 = False

                if visualized:
                    cv2.circle(self.processing_image, (x_idx, y_idx), 15, (0, 255, 255), cv2.FILLED)

            # Left click
            elif self.raised_fingers[1] == True and self.raised_fingers[0] == True and self.raised_fingers.count(False) >= 3:
                if not self.is_clicked1:
                    mouse.click(button="left")

                if visualized:
                    x_tmb, y_tmb = self.scaled_landmarks_position[4][:-1]
                    if not self.is_clicked1:
                        cv2.circle(self.processing_image, (x_tmb, y_tmb), 15, (0, 255, 0), cv2.FILLED)
                    else:
                        cv2.circle(self.processing_image, (x_tmb, y_tmb), 15, (0, 0, 255), cv2.FILLED)

                self.is_clicked1 = True
            
            # Right click
            elif self.raised_fingers[1] == True and self.raised_fingers[4] == True and self.raised_fingers.count(False) >= 3:
                if not self.is_clicked1:
                    mouse.click(button="right")

                if visualized:
                    x_pnk, y_pnk = self.scaled_landmarks_position[20][:-1]
                    if not self.is_clicked1:
                        cv2.circle(self.processing_image, (x_pnk, y_pnk), 15, (0, 255, 0), cv2.FILLED)
                    else:
                        cv2.circle(self.processing_image, (x_pnk, y_pnk), 15, (0, 0, 255), cv2.FILLED)

                self.is_clicked1 = True

            # Drag & Drop, Double Click
            elif self.raised_fingers[1] == True and self.raised_fingers[2] == True:
                # Press and Release Left Mouse Button
                if self.raised_fingers.count(False) >= 3:
                    x_mid, y_mid = self.scaled_landmarks_position[12][:-1]

                    if self.hand_detector.isIndexMiddleTouching(self.main_hand):
                        if not self.is_clicked1:
                            mouse.press(button='left')
                        else:
                            moveCursor(x_mid, y_mid)

                        if visualized:
                            if not self.is_clicked1:
                                cv2.circle(self.processing_image, (x_mid, y_mid), 15, (0, 255, 0), cv2.FILLED)
                            else:
                                cv2.circle(self.processing_image, (x_mid, y_mid), 15, (0, 0, 255), cv2.FILLED)

                        self.is_clicked1 = True
                    
                    else:
                        mouse.release()

                        moveCursor(x_mid, y_mid)

                        self.is_clicked1 = False

                        if visualized:
                            cv2.circle(self.processing_image, (x_mid, y_mid), 15, (0, 255, 255), cv2.FILLED)

                # Press and Release Right Mouse Button
                elif self.raised_fingers[4] == True and self.raised_fingers.count(False) >= 2:
                    x_mid, y_mid = self.scaled_landmarks_position[12][:-1]

                    if self.hand_detector.isIndexMiddleTouching(self.main_hand):
                        if not self.is_clicked1:
                            mouse.press(button='right')
                        else:
                            moveCursor(x_mid, y_mid)

                        if visualized:
                            if not self.is_clicked1:
                                cv2.circle(self.processing_image, (x_mid, y_mid), 15, (0, 255, 0), cv2.FILLED)
                            else:
                                cv2.circle(self.processing_image, (x_mid, y_mid), 15, (0, 0, 255), cv2.FILLED)

                        self.is_clicked1 = True
                    
                    else:
                        mouse.release()

                        moveCursor(x_mid, y_mid)

                        self.is_clicked1 = False

                        if visualized:
                            cv2.circle(self.processing_image, (x_mid, y_mid), 15, (0, 255, 255), cv2.FILLED)

                # Double Click
                elif self.raised_fingers[3] == True and self.raised_fingers.count(False) >= 2:
                    mouse.release()

                    x_mid, y_mid = self.scaled_landmarks_position[12][:-1]

                    if self.hand_detector.isIndexMiddleTouching(self.main_hand) and self.hand_detector.isMiddleRingTouching(self.main_hand):
                        if not self.is_clicked1:
                            mouse.double_click(button='left')

                        if visualized:
                            if not self.is_clicked1:
                                cv2.circle(self.processing_image, (x_mid, y_mid), 15, (0, 255, 0), cv2.FILLED)
                            else:
                                cv2.circle(self.processing_image, (x_mid, y_mid), 15, (0, 0, 255), cv2.FILLED)

                        self.is_clicked1 = True
                    
                    else:
                        moveCursor(x_mid, y_mid)

                        self.is_clicked1 = False

                        if visualized:
                            cv2.circle(self.processing_image, (x_mid, y_mid), 15, (0, 255, 255), cv2.FILLED)

            # Scroll Up
            # elif self.raised_fingers[4] == True and self.raised_fingers.count(False) >= 4:
            #     mouse.scroll(0, 1)

            #     if visualized:
            #         x_pnk, y_pnk = self.scaled_landmarks_position[20][:-1]
            #         cv2.circle(self.processing_image, (x_pnk, y_pnk), 15, (0, 255, 0), cv2.FILLED)

            # Scroll Down
            # elif self.flipped_raised_fingers[4] == True and self.flipped_raised_fingers.count(False) >= 4:
            #     mouse.scroll(0, -1)

            #     if visualized:
            #         x_pnk, y_pnk = self.scaled_landmarks_position[20][:-1]
            #         cv2.circle(self.processing_image, (x_pnk, y_pnk), 15, (0, 255, 0), cv2.FILLED)

            else:
                mouse.release()
                self.is_clicked1 = False

        if visualized:
            return self.processing_image

    def endControl(self):
        if self.is2HandControlActivated() and self.raised_fingers[4] == True and self.raised_fingers.count(False) >= 4 and self.other_raised_fingers[4] == True and self.other_raised_fingers.count(False) >= 4:
            return True
        return False

wCam, hCam = 800, 450

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

monitor = screeninfo.get_monitors()[0]
wScr, hScr = monitor.width, monitor.height

gesture_mouse_control = GestureMouseControl(htm.HandType.RIGHT_HAND.value, wCam, hCam, wScr, hScr, click_threshold=0.03)

while True:
    success, img = cap.read()

    img = cv2.flip(img, 1)

    gesture_mouse_control.process(img)

    img = gesture_mouse_control.visualize(flipped=False)

    img = gesture_mouse_control.controlCursor(visualized=True, flipped=False)

    img = gesture_mouse_control.extendedControl(visualized=True)

    fps = htm.getFPS()

    cv2.putText(img, "FPS: " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    cv2.imshow("Laptop Webcam", img)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or gesture_mouse_control.endControl():
        break
    
cap.release()
cv2.destroyAllWindows()