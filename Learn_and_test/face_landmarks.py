## Step 1 IMPORT AND INITIALISE FACEMESH
#The Imports--------------------------
import cv2
import mediapipe as mp 
import numpy as np 

import time # this is step 6 prt 1

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

#step 5 prt 1 : frame counter 
frame_count = 0

#step 6 prt 2 
prev_time = time.time()


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
                cv2.circle(frame, (x_px,y_px), 1, (0,255,0), -1)
            
            #Step 4: 6 eye landmarks - red, larger
            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            for idx in LEFT_EYE:
                lm = face_lms.landmark[idx]
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                cv2.circle(frame, (x_px, y_px,), 3, (0,0,255), -1)
            #----------------------------------------------
    
    #step 5 prt 2 : get 6 coordinate pairs and left eye squint
    frame_count += 1 
    if frame_count % 30 == 0 and results.multi_face_landmarks:
        print(f"\n---- Frame {frame_count} -----")
        face_lms = results.multi_face_landmarks[0]
        for i, idx in enumerate(LEFT_EYE):
            lm = face_lms.landmark[idx]
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            print(f" P{i+1} (lm {idx}): ({x_px}, {y_px})")
    #--------------------------------------------------------

    #step 6 prt 3: FPS and HUD 
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time + 1e-9))
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, "Landmarks: 468",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    if results.multi_face_landmarks:
        cv2.putText(frame, "Face: DETECTED",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        cv2.putText(frame, "Face: Not DETECTED",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)    
    

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

