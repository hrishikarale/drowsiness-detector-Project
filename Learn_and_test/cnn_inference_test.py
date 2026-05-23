import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

# Load model
model = tf.keras.models.load_model(r"models\eye_state_classifier.h5")
print("Model loaded.")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]

#Warming up the model 
dummy = np.zeros((1, 24, 24, 1), dtype=np.float32)
model.predict(dummy, verbose=0)
print("Model warmed up.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark

        # Get eye bounding box from landmarks
        xs = [int(lms[i].x * w) for i in LEFT_EYE_IDX]
        ys = [int(lms[i].y * h) for i in LEFT_EYE_IDX]

        x1, x2 = max(min(xs)-5, 0), min(max(xs)+5, w)
        y1, y2 = max(min(ys)-5, 0), min(max(ys)+5, h)

        eye_region = frame[y1:y2, x1:x2]

        if eye_region.size > 0:
            # Preprocess
            t_start = time.time()
            eye_small  = cv2.resize(eye_region, (24, 24))
            eye_grey   = cv2.cvtColor(eye_small, cv2.COLOR_BGR2GRAY)
            eye_input  = eye_grey.reshape(1, 24, 24, 1).astype(np.float32) / 255.0
            eye_tensor = tf.constant(eye_input, dtype=tf.float32)
            prediction = float(model(eye_tensor, training=False)[0][0])
            t_end      = time.time()

            inference_ms = (t_end - t_start) * 1000
            state        = "OPEN" if prediction > 0.5 else "CLOSED"

            print(f"Prediction: {prediction:.3f} | State: {state} | Time: {inference_ms:.1f}ms")

            # Draw on frame
            color = (0,255,0) if prediction > 0.5 else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{state} {prediction:.2f}",
                        (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("CNN Inference Test", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()