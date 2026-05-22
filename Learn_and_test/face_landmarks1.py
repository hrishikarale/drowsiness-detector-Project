import cv2
import mediapipe as mp
import numpy as np
import time
from ear_calculator import get_eye_coords, calculate_EAR

# ── MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("FaceMesh initialised successfully")

# ── Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: can't open webcam")
    exit()

# ── Canvas and camera dimensions
CANVAS_W, CANVAS_H = 1280, 720
CAM_W,    CAM_H    = 640,  480
CAM_X = (CANVAS_W - CAM_W) // 2
CAM_Y = (CANVAS_H - CAM_H) // 2

# ── Colours (all BGR)
CYAN    = (255, 220, 0)
RED     = (0,   0,   220)
ORANGE  = (0,   165, 255)
WHITE   = (255, 255, 255)
DIM     = (80,  80,  80)

# ── Eye and face indices
LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
FACE_OVAL     = [10,338,297,332,284,251,389,356,454,323,
                 361,288,397,365,379,378,400,377,152,148,
                 176,149,150,136,172,58,132,93,234,127,
                 162,21,54,103,67,109,10]
NOSE_BRIDGE   = [168, 6, 197, 195, 4, 1]
L_BROW        = [70, 63, 105, 66, 107]
R_BROW        = [336, 296, 334, 293, 300]
LIPS_OUTER    = [61,185,40,39,37,0,267,269,270,409,291,
                 375,321,405,314,17,84,181,91,146,61]

# ── Thresholds
EAR_THRESHOLD = 0.25
FRAME_LIMIT   = 20

# ── State
closed_frames = 0
prev_time     = time.time()
ear           = 0.0
scan_y        = 0
scan_dir      = 1
SCAN_SPEED    = 6

# ── Create resizable window
cv2.namedWindow("DROWSINESS DETECTOR", cv2.WINDOW_NORMAL)
cv2.resizeWindow("DROWSINESS DETECTOR", CANVAS_W, CANVAS_H)


# ── Helpers
def draw_structure(frame, landmarks, indices, w, h, color):
    coords = get_eye_coords(landmarks, indices, w, h)
    pts = np.array(coords, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], False, color, 1)


def draw_corner_brackets(canvas, x, y, w, h, color, size=24, thick=2):
    cv2.line(canvas, (x, y),     (x+size, y),   color, thick)
    cv2.line(canvas, (x, y),     (x, y+size),   color, thick)
    cv2.line(canvas, (x+w, y),   (x+w-size, y), color, thick)
    cv2.line(canvas, (x+w, y),   (x+w, y+size), color, thick)
    cv2.line(canvas, (x, y+h),   (x+size, y+h), color, thick)
    cv2.line(canvas, (x, y+h),   (x, y+h-size), color, thick)
    cv2.line(canvas, (x+w, y+h), (x+w-size, y+h), color, thick)
    cv2.line(canvas, (x+w, y+h), (x+w, y+h-size), color, thick)


def draw_hud_panel(canvas, x, y, w, h, lines, title=None):
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0,15,10), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), CYAN, 1)
    if title:
        cv2.putText(canvas, title, (x+8, y+16),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)
        cv2.line(canvas, (x, y+20), (x+w, y+20), CYAN, 1)
    for i, (label, val, col) in enumerate(lines):
        ty = y + (30 if title else 16) + i*22
        cv2.putText(canvas, label, (x+8, ty),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)
        cv2.putText(canvas, str(val), (x+90, ty),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)


# ── Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w    = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            lms = face_lms.landmark

            draw_structure(frame, lms, FACE_OVAL,   w, h, DIM)
            draw_structure(frame, lms, NOSE_BRIDGE, w, h, DIM)
            draw_structure(frame, lms, L_BROW,      w, h, DIM)
            draw_structure(frame, lms, R_BROW,      w, h, DIM)
            draw_structure(frame, lms, LIPS_OUTER,  w, h, DIM)

            left_coords  = get_eye_coords(lms, LEFT_EYE_IDX,  w, h)
            right_coords = get_eye_coords(lms, RIGHT_EYE_IDX, w, h)
            ear = (calculate_EAR(left_coords) + calculate_EAR(right_coords)) / 2.0

            eye_color = RED if ear < EAR_THRESHOLD else (0,255,0)
            for coords in [left_coords, right_coords]:
                pts = np.array(coords, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True,
                             (min(eye_color[0]+60,255),
                              min(eye_color[1]+60,255),
                              min(eye_color[2]+60,255)), 1)
                cv2.polylines(frame, [pts], True, eye_color, 1)

            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                closed_frames = 0

    # Build canvas
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = frame

    # Scan line
    scan_abs = CAM_Y + scan_y
    cv2.line(canvas, (CAM_X, scan_abs), (CAM_X+CAM_W, scan_abs), (0,200,180), 1)
    fade = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    cv2.line(fade, (CAM_X, scan_abs-1), (CAM_X+CAM_W, scan_abs-1), (0,80,70), 1)
    canvas = cv2.addWeighted(canvas, 1.0, fade, 0.5, 0)

    scan_y += SCAN_SPEED * scan_dir
    if scan_y >= CAM_H: scan_dir = -1
    if scan_y <= 0:     scan_dir =  1

    # Corner brackets
    bracket_col = RED if closed_frames >= FRAME_LIMIT else \
                  ORANGE if closed_frames > 0 else CYAN
    draw_corner_brackets(canvas, CAM_X, CAM_Y, CAM_W, CAM_H,
                         bracket_col, size=30, thick=2)
    cv2.rectangle(canvas, (CAM_X, CAM_Y),
                 (CAM_X+CAM_W, CAM_Y+CAM_H), (0,60,50), 1)

    # Left HUD panel
    curr_time = time.time()
    fps       = int(1 / (curr_time - prev_time + 1e-9))
    prev_time = curr_time
    ear_col   = RED if ear < EAR_THRESHOLD else CYAN

    draw_hud_panel(canvas, 20, CAM_Y, 190, 120,
        title="SYS METRICS",
        lines=[
            ("FPS",     fps,                         CYAN),
            ("EAR",     f"{ear:.3f}",                ear_col),
            ("COUNTER", f"{closed_frames}/{FRAME_LIMIT}", WHITE),
        ])

    # Right HUD panel
    face_status = "DETECTED"  if results.multi_face_landmarks else "NOT FOUND"
    face_col    = CYAN         if results.multi_face_landmarks else RED
    state_str   = "ALERT"     if closed_frames >= FRAME_LIMIT else \
                  "WARNING"   if closed_frames >  0            else "ACTIVE"
    state_col   = RED          if closed_frames >= FRAME_LIMIT else \
                  ORANGE       if closed_frames >  0            else CYAN

    draw_hud_panel(canvas, CANVAS_W-210, CAM_Y, 190, 120,
        title="SYS STATUS",
        lines=[
            ("FACE",  face_status,  face_col),
            ("STATE", state_str,    state_col),
            ("THR",   EAR_THRESHOLD, WHITE),
        ])

    # Title bar
    cv2.putText(canvas, "DRIVER DROWSINESS DETECTION SYSTEM",
               (320, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 1)
    cv2.line(canvas, (20, 50), (CANVAS_W-20, 50), (0,80,60), 1)

    # Bottom alert / status bar
    if closed_frames >= FRAME_LIMIT:
        cv2.rectangle(canvas, (0, CANVAS_H-50), (CANVAS_W, CANVAS_H), (0,0,160), -1)
        cv2.putText(canvas, "DROWSINESS DETECTED — PLEASE TAKE A BREAK",
                   (220, CANVAS_H-18), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    else:
        cv2.line(canvas, (20, CANVAS_H-20), (CANVAS_W-20, CANVAS_H-20), (0,80,60), 1)
        cv2.putText(canvas,
                   "MONITORING ACTIVE  |  EAR THRESHOLD: 0.25  |  FRAME LIMIT: 20",
                   (300, CANVAS_H-6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,120,100), 1)

    cv2.imshow("DROWSINESS DETECTOR", canvas)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()