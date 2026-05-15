import cv2
import mediapipe as mp

# -------------------------------
# MediaPipe setup
# -------------------------------
mp_hands = mp.solutions.hands
#access MediaPipe's hand tracking model
#Contains: hand detector, 21 hand landmark model, connection structure(bones)

mp_draw = mp.solutions.drawing_utils
#Utility to draw: landmarks(dots), connections(lines between joints)
#---------------------------------

hands = mp_hands.Hands(
    max_num_hands=2,
    #Detects upto two hands
    min_detection_confidence=0.7
    #Model must be > 70% confident before accepting detection
)


# -------------------------------
# Open DroidCam (CHANGE INDEX if needed)
# -------------------------------
camera_index = 1  # usually 1, 2 or 3 for DroidCam
#in this case it opens camera device index 1
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
#Uses MSFS backend(Windows Media Foundation)
#OpenCV doesn't know Phone camera, it only sees: 0 -> laptop webcam, 1 -> DroidCam virtual camera(your phone)  

#--------------------------------

if not cap.isOpened():
    print("Camera not opened")
    exit()
#Ensures camera is sucessfully opened

# -------------------------------
# Main loop
# -------------------------------
while True:#Runs forever

    ret, frame = cap.read()
    #ret=True/False(frame sucess)  frame= actual image(NumPy array)

    if not ret:
        print("Failed to access webcam")
        break
    #stops if camera disconnets, stream breaks

    h, w = frame.shape[:2]
    # h=height(rows), w=width(coloums)
    #used to convert normalised coordinates -> pixels

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #Color Conversion: OpenCV uses BGR format, MediaPipe expects RGB

    results = hands.process(rgb)
    #frame is passed to ML model, model outputs: detected hands, 21 landmarks per hand

    if results.multi_hand_landmarks:
    #if at least one hand is detected -> True Otherwise skip block

        for hand_landmarks in results.multi_hand_landmarks:
        #Why Loop? Because multiple hands possible(upto 2 in this code)
        #each hand_landmark contains: 21 points(fingers, wrist,etc)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            #Draws: dots(landmarks), lines(bone connections)

            wrist = hand_landmarks.landmark[0]
            #Landmark 0 = wrist point, all landmarks are indexed 0-20

            px = int(wrist.x * w)
            py = int(wrist.y * h)
            #MediaPipe give:
            #  wrist.x -> 0 to 1(relative position)
            #  wrist.y -> 0 to 1
            # pixel_x = normalised_x * image_width
            # pixel_y = normalised_y * image_height

            cv2.circle(frame, (px, py), 10, (0, 255, 0), -1)
            #draw circles on Wrist
            #(px, py): position
            #10 : radius
            #(0,255,0) : green color(BGR)
            #-1: filled circle

            #This function Draws text on video frames
            cv2.putText(
                frame,     #The image you are drawing on
                f'wrist: ({px}, {py})', #Text to display
                (px + 12, py),   #Position of text
                cv2.FONT_HERSHEY_SIMPLEX,    #Font type
                0.5,     #Font scale (size)
                (255, 255, 255),   #Color (BGR format)
                1    #Color (BGR format)
            )
            

            print(f'Wrist position: {px}, {py}')
            #Printing to terminal: Debugging, Se real time coordinates in consol

    cv2.imshow("DroidCam Hand Tracking", frame)
    #Displaying Output Frame: Opens window, Show Processed video stream, with skeleton+overlays

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #Exit condition: Waits 1ms per frame, If q pressed exits loop

cap.release()
cv2.destroyAllWindows()
#Cleanup : Rleases camera resource, Closes OpenCV windows, Prevent Camera lock issues

#the program cycle:
# Phone camera → OpenCV frame → MediaPipe hand model →
# 21 landmarks → wrist extraction → pixel mapping →
# drawing + display + real-time tracking