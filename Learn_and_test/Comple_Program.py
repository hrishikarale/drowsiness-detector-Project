import cv2
import mediapipe as mp
import numpy as np

# Setup
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb  = frame[:, :, ::-1]
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            for lm in hand_lm.landmark:
                px = int(lm.x * w)
                py = int(lm.y * h)
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            tip = hand_lm.landmark[8]
            cv2.putText(frame, 'Index tip', (int(tip.x*w)+8, int(tip.y*h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.imshow('Hand tracking', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()