import cv2
import mediapipe as mp
import numpy as np
import time
import math
from ear_calculator import get_eye_coords, calculate_EAR

# ── MediaPipe setup ───────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("J.A.R.V.I.S initialising...")
print("FaceMesh loaded successfully.")

# ── Webcam ────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: can't open webcam")
    exit()

# ── Canvas and camera dimensions ─────────────────────
CANVAS_W, CANVAS_H = 1280, 720
CAM_W,    CAM_H    = 640,  480
CAM_X = (CANVAS_W - CAM_W) // 2   # 320 — centre horizontally
CAM_Y = (CANVAS_H - CAM_H) // 2   # 120 — centre vertically

# ── Colours (BGR) ────────────────────────────────────
CYAN      = (255, 220,   0)   # bright cyan
CYAN_DIM  = (120, 100,   0)   # dim cyan for outlines
RED       = (  0,  30, 220)   # alert red
RED_DIM   = (  0,  15, 100)   # dim red
ORANGE    = (  0, 165, 255)   # warning orange
WHITE     = (255, 255, 255)
BLACK     = (  0,   0,   0)
PANEL_BG  = (  0,  20,  10)   # very dark green-tint for panels
DIM_GREY  = ( 60,  60,  60)   # grey mesh lines

# ── Eye and face indices ──────────────────────────────
LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

FACE_OVAL   = [10,338,297,332,284,251,389,356,454,323,
               361,288,397,365,379,378,400,377,152,148,
               176,149,150,136,172,58,132,93,234,127,
               162,21,54,103,67,109,10]
NOSE_BRIDGE = [168, 6, 197, 195, 4, 1]
L_BROW      = [70,  63, 105, 66, 107]
R_BROW      = [336, 296, 334, 293, 300]
LIPS_OUTER  = [61,185,40,39,37,0,267,269,270,409,291,
               375,321,405,314,17,84,181,91,146,61]

# ── Thresholds ────────────────────────────────────────
EAR_THRESHOLD = 0.25
FRAME_LIMIT   = 20

# ── State ─────────────────────────────────────────────
closed_frames = 0
prev_time     = time.time()
ear           = 0.0
frame_num     = 0

# Scan line state
scan_y    = 0
scan_dir  = 1
SCAN_SPD  = 5

# Rotating arc angle
arc_angle = 0.0

# Alert pulse
alert_alpha = 0

# ── Resizable window ──────────────────────────────────
cv2.namedWindow("J.A.R.V.I.S", cv2.WINDOW_NORMAL)
cv2.resizeWindow("J.A.R.V.I.S", CANVAS_W, CANVAS_H)


# ─────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────

def draw_structure(frame, lms, indices, w, h, color):
    """Draw a polyline through landmark indices — grey face structure."""
    coords = get_eye_coords(lms, indices, w, h)
    pts = np.array(coords, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], False, color, 1)


def draw_eye_glow(frame, coords, color):
    """Draw eye outline with a glow — two layers, thick dim then sharp."""
    pts = np.array(coords, np.int32).reshape((-1, 1, 2))
    # glow layer — thicker, dimmer
    glow = tuple(min(c + 40, 255) for c in color)
    cv2.polylines(frame, [pts], True, glow, 4)
    # sharp layer on top
    cv2.polylines(frame, [pts], True, color, 1)
    # dot at each landmark
    for x, y in coords:
        cv2.circle(frame, (x, y), 2, color, -1)


def draw_bracket(canvas, x, y, w, h, color, size=20, thick=2):
    """L-shaped corner brackets around the camera feed."""
    corners = [
        (x,     y,      1,  1),
        (x + w, y,     -1,  1),
        (x,     y + h,  1, -1),
        (x + w, y + h, -1, -1),
    ]
    for cx, cy, dx, dy in corners:
        cv2.line(canvas, (cx, cy), (cx + dx * size, cy),         color, thick)
        cv2.line(canvas, (cx, cy), (cx,             cy + dy * size), color, thick)


def draw_rotating_arc(canvas, cx, cy, radius, angle, color, span=1.2, thick=1):
    """Draw a rotating partial arc — the targeting circle."""
    axes = (radius, radius)
    start_deg = math.degrees(angle) % 360
    end_deg   = (start_deg + math.degrees(span)) % 360
    cv2.ellipse(canvas, (cx, cy), axes, 0,
                start_deg, end_deg, color, thick)


def draw_panel(canvas, x, y, w, h, title, rows, alert=False):
    """
    Semi-transparent HUD panel with cyan border and data rows.
    rows = list of (label, value, color) tuples.
    """
    col = RED if alert else CYAN

    # dark background — blend with existing canvas
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # border
    cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 1)

    # title bar
    cv2.line(canvas, (x, y + 18), (x + w, y + 18), col, 1)
    cv2.putText(canvas, title,
                (x + 6, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    # corner ticks on panel
    tick = 6
    for px, py in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
        sx = 1 if px == x else -1
        sy = 1 if py == y else -1
        cv2.line(canvas, (px, py), (px + sx * tick, py), col, 1)
        cv2.line(canvas, (px, py), (px, py + sy * tick), col, 1)

    # data rows
    for i, (label, value, vcol) in enumerate(rows):
        ry = y + 30 + i * 18
        cv2.putText(canvas, label,
                    (x + 6, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (100, 180, 160), 1)
        cv2.putText(canvas, str(value),
                    (x + w - 6, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    vcol, 1,
                    cv2.LINE_AA)


def draw_ear_bar(canvas, x, y, w, h, ear_val, threshold, alert):
    """Horizontal EAR progress bar below the camera feed."""
    col = RED if alert else CYAN
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 1)

    # fill
    fill_w = int((w - 4) * min(ear_val / 0.45, 1.0))
    fill_col = RED if ear_val < threshold else CYAN
    cv2.rectangle(canvas, (x + 2, y + 2),
                  (x + 2 + fill_w, y + h - 2), fill_col, -1)

    # threshold marker
    thresh_x = x + 2 + int((w - 4) * (threshold / 0.45))
    cv2.line(canvas, (thresh_x, y), (thresh_x, y + h), WHITE, 1)

    # labels
    cv2.putText(canvas, f"EAR: {ear_val:.3f}",
                (x + 6, y + h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, WHITE, 1)
    cv2.putText(canvas, f"THR:{threshold}",
                (x + w - 55, y + h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                (180, 180, 180), 1)


def draw_scan_line(canvas, cam_x, cam_y, cam_w, sy, alert):
    """Horizontal scan line sweeping inside the camera feed."""
    col = RED if alert else (180, 230, 0)
    abs_y = cam_y + sy
    cv2.line(canvas,
             (cam_x,          abs_y),
             (cam_x + cam_w,  abs_y),
             col, 1)
    # subtle fade beneath
    fade = np.zeros_like(canvas)
    fade_h = 20
    for i in range(fade_h):
        alpha = int(40 * (1 - i / fade_h))
        cv2.line(fade,
                 (cam_x,         abs_y + i),
                 (cam_x + cam_w, abs_y + i),
                 (0, alpha, int(alpha * 0.8)), 1)
    cv2.add(canvas, fade, canvas)


def draw_ticker_dots(canvas, cx, cy, radius, count=12, angle_offset=0, col=None):
    """Small tick marks around a circle — like a radar dial."""
    if col is None:
        col = CYAN_DIM
    for i in range(count):
        a = angle_offset + (2 * math.pi * i / count)
        r1 = radius - (4 if i % 3 == 0 else 2)
        r2 = radius
        x1 = int(cx + math.cos(a) * r1)
        y1 = int(cy + math.sin(a) * r1)
        x2 = int(cx + math.cos(a) * r2)
        y2 = int(cy + math.sin(a) * r2)
        cv2.line(canvas, (x1, y1), (x2, y2), col, 1)


# ─────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    h, w    = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # ── Process landmarks ─────────────────────────────
    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            lms = face_lms.landmark

            # Grey structural mesh
            draw_structure(frame, lms, FACE_OVAL,   w, h, DIM_GREY)
            draw_structure(frame, lms, NOSE_BRIDGE, w, h, DIM_GREY)
            draw_structure(frame, lms, L_BROW,      w, h, DIM_GREY)
            draw_structure(frame, lms, R_BROW,      w, h, DIM_GREY)
            draw_structure(frame, lms, LIPS_OUTER,  w, h, DIM_GREY)

            # EAR calculation
            left_coords  = get_eye_coords(lms, LEFT_EYE_IDX,  w, h)
            right_coords = get_eye_coords(lms, RIGHT_EYE_IDX, w, h)
            ear = (calculate_EAR(left_coords) + calculate_EAR(right_coords)) / 2.0

            # Eye contours with glow
            eye_color = RED if ear < EAR_THRESHOLD else CYAN
            draw_eye_glow(frame, left_coords,  eye_color)
            draw_eye_glow(frame, right_coords, eye_color)

            # Frame counter
            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                closed_frames = 0

    alert = closed_frames >= FRAME_LIMIT

    # ── Build black canvas ────────────────────────────
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # Paste camera feed centred
    canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = frame

    # ── Scan line ─────────────────────────────────────
    draw_scan_line(canvas, CAM_X, CAM_Y, CAM_W, scan_y, alert)
    scan_y += SCAN_SPD * scan_dir
    if scan_y >= CAM_H: scan_dir = -1
    if scan_y <= 0:     scan_dir =  1

    # ── Rotating targeting arcs ───────────────────────
    face_cx = CAM_X + CAM_W // 2
    face_cy = CAM_Y + CAM_H // 2 - 20
    radius  = int(CAM_H * 0.28)

    arc_col   = RED if alert else CYAN
    arc_col2  = RED_DIM if alert else CYAN_DIM

    # outer slow arc
    draw_rotating_arc(canvas, face_cx, face_cy,
                      radius + 10, -arc_angle * 0.4,
                      arc_col2, span=1.0, thick=1)
    # inner fast arc
    draw_rotating_arc(canvas, face_cx, face_cy,
                      radius, arc_angle,
                      arc_col, span=1.4, thick=1)
    # counter arc
    draw_rotating_arc(canvas, face_cx, face_cy,
                      radius, -arc_angle * 0.7 + math.pi,
                      arc_col2, span=0.6, thick=1)

    # tick marks
    draw_ticker_dots(canvas, face_cx, face_cy,
                     radius + 10, count=24,
                     angle_offset=arc_angle * 0.3,
                     col=CYAN_DIM if not alert else RED_DIM)

    arc_angle += 0.018

    # ── Corner brackets (state-coloured) ─────────────
    bracket_col = RED if alert else (ORANGE if closed_frames > 0 else CYAN)
    draw_bracket(canvas, CAM_X, CAM_Y, CAM_W, CAM_H,
                 bracket_col, size=22, thick=2)

    # dim feed border
    cv2.rectangle(canvas,
                  (CAM_X, CAM_Y),
                  (CAM_X + CAM_W, CAM_Y + CAM_H),
                  CYAN_DIM, 1)

    # ── Data lines from face to panels ───────────────
    line_col = (40, 80, 60)
    cv2.line(canvas,
             (CAM_X, CAM_Y + int(CAM_H * 0.3)),
             (CAM_X - 10, CAM_Y + int(CAM_H * 0.3)),
             CYAN_DIM, 1)
    cv2.line(canvas,
             (CAM_X + CAM_W, CAM_Y + int(CAM_H * 0.3)),
             (CAM_X + CAM_W + 10, CAM_Y + int(CAM_H * 0.3)),
             CYAN_DIM, 1)

    # ── FPS ───────────────────────────────────────────
    curr_time = time.time()
    fps       = int(1 / (curr_time - prev_time + 1e-9))
    prev_time = curr_time

    # ── LEFT HUD panel ────────────────────────────────
    ear_col   = RED if ear < EAR_THRESHOLD else CYAN
    state_str = "ALERT"   if alert else \
                "WARNING" if closed_frames > 0 else "ACTIVE"
    state_col = RED if alert else (ORANGE if closed_frames > 0 else CYAN)

    draw_panel(canvas, 12, CAM_Y, 180, 120,
               "SYS.METRICS",
               [
                   ("FPS",     f"{fps}",                    CYAN),
                   ("EAR",     f"{ear:.3f}",                ear_col),
                   ("COUNTER", f"{closed_frames}/{FRAME_LIMIT}", WHITE),
                   ("STATE",   state_str,                   state_col),
               ],
               alert=alert)

    # ── RIGHT HUD panel ───────────────────────────────
    face_txt = "DETECTED"  if results.multi_face_landmarks else "NOT FOUND"
    face_col = CYAN        if results.multi_face_landmarks else RED
    eye_txt  = "CLOSED"    if ear < EAR_THRESHOLD else "OPEN"
    eye_col  = RED         if ear < EAR_THRESHOLD else CYAN

    draw_panel(canvas, CANVAS_W - 192, CAM_Y, 180, 120,
               "BIO.STATUS",
               [
                   ("FACE",    face_txt,   face_col),
                   ("EYES",    eye_txt,    eye_col),
                   ("SCAN",    "RUNNING",  CYAN),
                   ("THRESH",  f"{EAR_THRESHOLD}", WHITE),
               ],
               alert=alert)

    # ── Small top panels ─────────────────────────────
    # top-left
    cv2.rectangle(canvas, (12, 8), (12 + 160, 38), PANEL_BG, -1)
    cv2.rectangle(canvas, (12, 8), (12 + 160, 38), CYAN_DIM, 1)
    cv2.putText(canvas, "DRIVER ANALYSIS",
                (18, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)
    cv2.putText(canvas, "BIOMETRIC SCAN  ACTIVE",
                (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                (100, 200, 160), 1)

    # top-right — clock
    now = time.localtime()
    clock_str = f"{now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}"
    cv2.rectangle(canvas, (CANVAS_W - 172, 8), (CANVAS_W - 12, 38),
                  PANEL_BG, -1)
    cv2.rectangle(canvas, (CANVAS_W - 172, 8), (CANVAS_W - 12, 38),
                  CYAN_DIM, 1)
    cv2.putText(canvas, clock_str,
                (CANVAS_W - 166, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
    cv2.putText(canvas, "SYSTEM ONLINE",
                (CANVAS_W - 166, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                (100, 200, 160), 1)

    # ── EAR bar below camera feed ─────────────────────
    draw_ear_bar(canvas,
                 CAM_X, CAM_Y + CAM_H + 6,
                 CAM_W, 20,
                 ear, EAR_THRESHOLD, alert)

    # ── Title bar ─────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, 48), PANEL_BG, -1)
    cv2.line(canvas, (0, 48), (CANVAS_W, 48), CYAN_DIM, 1)
    cv2.putText(canvas,
                "J.A.R.V.I.S  —  DRIVER DROWSINESS DETECTION SYSTEM",
                (CANVAS_W // 2 - 260, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1)
    # title corner ticks
    for tx, ty in [(0, 0), (CANVAS_W - 1, 0)]:
        dx = 1 if tx == 0 else -1
        cv2.line(canvas, (tx, ty), (tx + dx * 20, ty), CYAN, 2)
        cv2.line(canvas, (tx, ty), (tx,           ty + 14), CYAN, 2)

    # ── Bottom alert / status bar ─────────────────────
    if alert:
        # pulsing red bar
        pulse = abs(math.sin(frame_num * 0.08))
        alpha = int(180 + pulse * 75)
        bar = canvas.copy()
        cv2.rectangle(bar, (0, CANVAS_H - 50), (CANVAS_W, CANVAS_H),
                      (0, 0, alpha), -1)
        cv2.addWeighted(bar, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas,
                 (0, CANVAS_H - 50),
                 (CANVAS_W, CANVAS_H - 50),
                 RED, 1)
        cv2.putText(canvas,
                    "DROWSINESS DETECTED  —  PLEASE TAKE A BREAK",
                    (CANVAS_W // 2 - 260, CANVAS_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)
    else:
        cv2.rectangle(canvas, (0, CANVAS_H - 22), (CANVAS_W, CANVAS_H),
                      PANEL_BG, -1)
        cv2.line(canvas,
                 (0, CANVAS_H - 22),
                 (CANVAS_W, CANVAS_H - 22),
                 CYAN_DIM, 1)
        cv2.putText(canvas,
                    "MONITORING ACTIVE  |  EAR THRESHOLD: 0.25"
                    "  |  FRAME LIMIT: 20  |  STATUS: NOMINAL",
                    (CANVAS_W // 2 - 310, CANVAS_H - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (60, 140, 100), 1)

    # ── Show ──────────────────────────────────────────
    cv2.imshow("J.A.R.V.I.S", canvas)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
