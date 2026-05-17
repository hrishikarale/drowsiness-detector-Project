import cv2
import mediapipe as mp
import time

# -----------------------------------
# MediaPipe setup
# -----------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------------
# Camera
# -----------------------------------
cap = cv2.VideoCapture(0)

# FPS variables
prev_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb)

    # -----------------------------------
    # FACE DETECTION
    # -----------------------------------
    if results.multi_face_landmarks:

        for face_lms in results.multi_face_landmarks:

            # -----------------------------------
            # Draw cool face mesh
            # -----------------------------------
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_lms,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=
                mp_draw.DrawingSpec(
                    color=(0, 255, 255),   # neon yellow
                    thickness=1,
                    circle_radius=1
                )
            )

            # -----------------------------------
            # Eye landmark
            # -----------------------------------
            lm = face_lms.landmark[159]

            px = int(lm.x * w)
            py = int(lm.y * h)

            # Glow effect
            cv2.circle(frame, (px, py), 12, (0, 255, 255), 2)
            cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)

            # Coordinates text
            cv2.putText(
                frame,
                f"Eye: ({px}, {py})",
                (px + 15, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    # -----------------------------------
    # FPS Counter
    # -----------------------------------
    current_time = time.time()

    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # -----------------------------------
    # Status Text
    # -----------------------------------
    cv2.putText(
        frame,
        "FaceMesh AI Tracking",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # -----------------------------------
    # Show frame
    # -----------------------------------
    cv2.imshow("AI Face Tracking", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()