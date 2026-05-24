## Step 1 IMPORT AND INITIALISE FACEMESH
#The Imports--------------------------
import cv2
import mediapipe as mp 
import numpy as np 
import pygame
import threading

import time # this is step 6 prt 1

from ear_calculator import get_eye_coords, calculate_EAR, calculate_MAR, MOUTH_IDX, MOUTH_OUTLINE

#TF imports
import tensorflow as tf

#-------------------------------------

#Initialise MediaPipe (outside the loop)------
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("FaceMesh initialised sucessfully")
#----------------------------------------------


#Step 2: THE WEBCAM LOOP (webcam open here)
#Open webcam --------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: can't open webcam")
    exit()
#-------------------------------------------

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

#Threashold value and counter for aleart
EAR_THRESHOLD = 0.25
closed_frames = 0
MAR_THRESHOLD = 0.53
yawn_frames = 0
yawn_timestamps = []
yawns_in_progress = False
yawns_per_min = 0


# #step 5 prt 1 : frame counter 
# frame_count = 0

#step 6 prt 2 
prev_time = time.time()

mar = 0.0 # initialising MAR

#Load CNN model ---------------------------
model = tf.keras.models.load_model(r"models\eye_state_classifier.h5")
# Warm up
dummy = np.zeros((1, 24, 24, 1), dtype=np.float32)
model.predict(dummy, verbose=0)
print("CNN model loaded and warmed up.")
#------------------------------------------------

# ── Audio setup ───────────────────────────────────────
def generate_beep(freq=1000, duration=0.5, sr=44100):
    t    = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    wave = (wave * 32767).astype(np.int16)
    wave = np.column_stack([wave, wave])
    return pygame.mixer.Sound(buffer=wave)

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
alarm_sound = generate_beep(freq=1000, duration=0.5)
alarm_sound.play()   # startup test beep — confirms audio works
print("Audio initialised — startup beep played.")
# ── Cooldown ──────────────────────────────────────────
last_alarm_time = 0
ALARM_COOLDOWN  = 3.0

#replacing Solid Rectangles
def draw_overlay(frame, x1, y1, x2, y2, color, alpha=0.5):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

#CNN state variable
cnn_closed_frames = 0

#The main loop-----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #Step 3 process frame + draw 468 dots
    #Send frame to mediapipe----------
    results = face_mesh.process(rgb)
    
    #-----------------------------
    #draw landmarks if face found----
    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            for lm in face_lms.landmark:
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                cv2.circle(frame, (x_px,y_px), 1, (50,50,50), -1)
            
            # #Step 4: 6 eye landmarks - red, larger
            # LEFT_EYE = [33, 160, 158, 133, 153, 144]
            # for idx in LEFT_EYE:
            #     lm = face_lms.landmark[idx]
            #     x_px = int(lm.x * w)
            #     y_px = int(lm.y * h)
            #     cv2.circle(frame, (x_px, y_px,), 3, (0,0,255), -1)
            #----------------------------------------------

            #calculate and display live EAR:
            left_coords = get_eye_coords(face_lms.landmark, LEFT_EYE_IDX, w, h)
            # print(f"Left eye coords: {left_coords}")
            right_coords = get_eye_coords(face_lms.landmark, RIGHT_EYE_IDX,w, h)
            left_ear = calculate_EAR(left_coords)
            right_ear = calculate_EAR(right_coords)
            ear = (left_ear + right_ear) / 2.0
            ear_color = (0,0,255) if ear < EAR_THRESHOLD else (0,255,0)
            eye_color = (0,0,255) if ear < EAR_THRESHOLD else (0,255,0)
            cv2.putText(frame, f"EAR: {ear:.2f}",
                        (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 1)

            for coords in [left_coords, right_coords]:
                pts = np.array(coords, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, eye_color, 1)

            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                closed_frames = 0
            cv2.putText(frame, f"Closed frames: {closed_frames}",
                        (10,125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            
            #CNN eye state detection
            xs = [int(face_lms.landmark[i].x*w) for i in LEFT_EYE_IDX]
            ys = [int(face_lms.landmark[i].y*h) for i in LEFT_EYE_IDX]
            x1, x2 = max(min(xs)-5, 0), min(max(xs)+5, w)
            y1, y2 = max(min(ys)-5, 0), min(max(ys)+5, h)

            eye_region = frame[y1:y2, x1:x2]

            if eye_region.size > 0:
                eye_small  = cv2.resize(eye_region, (24, 24))
                eye_grey   = cv2.cvtColor(eye_small, cv2.COLOR_BGR2GRAY)
                eye_input  = eye_grey.reshape(1, 24, 24, 1).astype(np.float32) / 255.0
                eye_tensor = tf.constant(eye_input, dtype=tf.float32)
                cnn_pred   = float(model(eye_tensor, training=False)[0][0])
                cnn_closed = cnn_pred < 0.5
                
                if cnn_closed:
                    cnn_closed_frames += 1
                else:
                    cnn_closed_frames = 0

                cnn_col = (0,0,255) if cnn_closed else (0,255,0)
                cv2.putText(frame, f"CNN: {'CLOSED' if cnn_closed else 'OPEN'} ({cnn_pred:.2f})",
                (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cnn_col, 1) 
                cv2.putText(frame, f"CNN frames: {cnn_closed_frames}/20",
                (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)




            #--------------------------------------------------
            #mouth coords and live value
            mouth_coords = get_eye_coords(face_lms.landmark, MOUTH_IDX, w, h)
            mar = calculate_MAR(mouth_coords)
            mar_color = (0,140,255) if mar > 0.55 else (0,255,255)
            cv2.putText(frame, f"MAR:{mar:.2f}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 1)
            
            #Yawn Counter
            if mar > MAR_THRESHOLD:
                yawn_frames += 1
            else:
                yawn_frames = 0
            cv2.putText(frame, f"Yawn: {yawn_frames}/15",
                        (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            if mar > MAR_THRESHOLD :
                mouth_outline_color = (0,0,255)
            elif mar > 0.40:
                mouth_outline_color = (0,165,255)
            else:
                mouth_outline_color = (0,255,0)
            mouth_outline_coords = get_eye_coords(
                face_lms.landmark,
                MOUTH_OUTLINE,
                w,
                h
            )
            mouth_pts = np.array(
                mouth_outline_coords,
                np.int32
            ).reshape((-1,1,2))
            cv2.polylines(
                frame, 
                [mouth_pts], 
                True, 
                mouth_outline_color, 
                1
            )
            
            #Yawn count per min
            if yawn_frames >= 15 and not yawns_in_progress:
                yawn_timestamps.append(time.time())
                yawns_in_progress = True
            elif yawn_frames == 0:
                yawns_in_progress = False
            
            now = time.time()
            yawn_timestamps = [t for t in yawn_timestamps if now - t <=60]
            yawns_per_min = len(yawn_timestamps)
            ypm_col = (0,0,255) if yawns_per_min >= 3 else (255,255,255)
            cv2.putText(frame, f"Yawns/min: {yawns_per_min}",
                        (10,175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ypm_col, 1)
            # if yawns_per_min >= 3:
            #     print("HIGH YAWN RATE - severe fatigue likely")


            

    # #step 5 prt 2 : get 6 coordinate pairs and left eye squint
    # frame_count += 1 
    # if frame_count % 30 == 0 and results.multi_face_landmarks:
    #     print(f"\n---- Frame {frame_count} -----")
    #     face_lms = results.multi_face_landmarks[0]
    #     for i, idx in enumerate(LEFT_EYE_IDX):
    #         lm = face_lms.landmark[idx]
    #         x_px = int(lm.x * w)
    #         y_px = int(lm.y * h)
    #         print(f" P{i+1} (lm {idx}): ({x_px}, {y_px})")
    # #--------------------------------------------------------
    

    #step 6 prt 3: FPS and HUD 
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time + 1e-9))
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    # cv2.putText(frame, "Landmarks: 468",
    #             (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    if results.multi_face_landmarks:
        cv2.putText(frame, "Face: DETECTED",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,0), 1)
    else:
        cv2.putText(frame, "Face: Not DETECTED",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)    
        
    if not results.multi_face_landmarks:
        cnn_closed_frames = 0

    eye_alert = (closed_frames >= 20) and (cnn_closed_frames >= 20)
    yawn_alert = yawn_frames >= 15
    high_yawn = yawns_per_min >= 3
    if high_yawn:
        alert_level = "CRITICAL"
    elif eye_alert and yawn_alert:
        alert_level = "COMPOUND"
    elif eye_alert:
        alert_level = "EYE"
    elif yawn_alert:
        alert_level = "YAWN"
    else: alert_level = "SAFE"


    #Solid Block 
    # if alert_level == "CRITICAL":
    #     cv2.rectangle(frame, (0,h-70), (w,h), (150,0,150), -1)
    #     cv2.putText(frame, f"CRITICAL - {yawns_per_min} YAWNS/MIN",
    #                 (60, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    #     cv2.rectangle(frame, (0,0), (w,h), (150,0,150), 3)

    # elif alert_level == "COMPOUND":
    #     cv2.rectangle(frame, (0,h-70), (w,h), (180,0,180), -1)
    #     cv2.putText(frame, "COMPOUND ALERT - EYES + YAWN",
    #                 (60,h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    #     cv2.rectangle(frame, (0,0), (w,h), (180,0,180), 3)

    # elif alert_level == "EYE":
    #     cv2.rectangle(frame, (0, h-70), (w,h), (0,0,200), -1)
    #     cv2.putText(frame, "DROWSY DETECTED",
    #                 (100,h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    #     cv2.rectangle(frame, (0,0), (w,h), (0,0,200), 3)

    # elif alert_level == "YAWN":
    #     cv2.rectangle(frame, (0,h-70), (w,h), (0,100,200), -1)
    #     cv2.putText(frame, "YAWN DETECTED",
    #                 (130,h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    #     cv2.rectangle(frame, (0,0), (w,h), (0,165,255), 3)
    
    # else:
    #     cv2.rectangle(frame, (0,0), (w,h), (0,200,0), 2)

    #Transparent Block
    if alert_level == "CRITICAL":
        draw_overlay(frame, 0, h-70, w, h, (150,0,150), alpha=0.7)
        cv2.putText(frame, f"CRITICAL - {yawns_per_min} YAWNS/MIN",
                    (160, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.rectangle(frame, (0,0), (w,h), (150,0,150), 3)

    elif alert_level == "COMPOUND":
        draw_overlay(frame, 0, h-70, w, h, (180,0,180), alpha=0.7)
        cv2.putText(frame, "COMPOUND ALERT - EYES + YAWN",
                    (60,h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.rectangle(frame, (0,0), (w,h), (180,0,180), 3)

    elif alert_level == "EYE":
        draw_overlay(frame, 0, h-70, w, h, (0,0,200), alpha=0.7)
        cv2.putText(frame, "DROWSY DETECTED",
                    (190,h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.rectangle(frame, (0,0), (w,h), (0,0,200), 3)

    elif alert_level == "YAWN":
        draw_overlay(frame, 0, h-70, w, h, (0,100,200), alpha=0.7)
        cv2.putText(frame, "YAWN DETECTED",
                    (200,h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.rectangle(frame, (0,0), (w,h), (0,165,255), 3)

    else:
        cv2.rectangle(frame, (0,0), (w,h), (0,200,0), 2)

    # ── Alarm with cooldown ───────────────────────────────
    if alert_level != "SAFE":
        now_t = time.time()
        if now_t - last_alarm_time > ALARM_COOLDOWN:
            threading.Thread(
                target=lambda: alarm_sound.play(),
                daemon=True
            ).start()
            last_alarm_time = now_t
    else:
        alarm_sound.stop()


    cv2.imshow('Face Landmarks', frame)
    if cv2.waitKey(1) == ord('q'):
        break
#--------------------------------------------

#Cleanup--------------------------------
cap.release()
cv2.destroyAllWindows()
#---------------------------------------

#Step 3:PROCESS FRAME + DRAW 468 DOTS
# Its added incide the loop

