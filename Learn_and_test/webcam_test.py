import cv2
import numpy as np

# Open the default camera (0 = first webcam)
cap = cv2.VideoCapture(0)

# Read ONE frame
ret, frame = cap.read()

# ret = True if it worked, False if camera failed
# frame = a NumPy array, shape (height, width, 3), dtype uint8

print(frame.shape) # e.g. (480, 640, 3)
print(frame.dtype) # uint8