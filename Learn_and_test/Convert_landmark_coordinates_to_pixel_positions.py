import cv2
import mediapipe as mp

# -------------------------------
# MediaPipe setup
# -------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

# -------------------------------
# Open DroidCam (CHANGE INDEX if needed)
# -------------------------------
camera_index = 2  # usually 1, 2 or 3 for DroidCam

cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

if not cap.isOpened():
    print("Camera not opened")
    exit()

# -------------------------------
# Main loop
# -------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to access webcam")
        break

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            wrist = hand_landmarks.landmark[0]

            px = int(wrist.x * w)
            py = int(wrist.y * h)

            cv2.circle(frame, (px, py), 10, (0, 255, 0), -1)

            cv2.putText(
                frame,
                f'wrist: ({px}, {py})',
                (px + 12, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            print(f'Wrist position: {px}, {py}')

    cv2.imshow("DroidCam Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()