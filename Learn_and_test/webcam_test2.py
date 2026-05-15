import cv2
import numpy as np

#Open the default Camera(0 = first webcam)
cap = cv2.VideoCapture(0)

while True: # loop forever
  h, w = frame.shape[:2]
  rgb = frame[:, :, ::-1]
  ret, frame = cap.read() # get the next frame
  if not ret: # camera failed? stop
    break

  # frame is your NumPy array — do stuff here

  cv2.imshow('JassiTheLassiDriver', frame) # show the frame
  if cv2.waitKey(1) == ord('q'): # press Q to quit, Waits 1ms between frames. Without this, the window freezes.
    break

cap.release() # free the camera
cv2.destroyAllWindows() # close the window