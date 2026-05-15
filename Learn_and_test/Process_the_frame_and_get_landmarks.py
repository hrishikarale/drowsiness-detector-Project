import cv2
import mediapipe as mp

# Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Get frame size (optional but useful later)
    h, w = frame.shape[:2]

    # BGR → RGB
    rgb = frame[:, :, ::-1]

    # Process frame
    results = hands.process(rgb)

    # If hand detected
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Wrist landmark (0)
            wrist = hand_landmarks.landmark[0]

            # Normalized coordinates (0 to 1)
            print(wrist.x, wrist.y)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()