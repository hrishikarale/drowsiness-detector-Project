import numpy as np

def euclidean_distance(point_a, point_b):
    a = np.array(point_a)
    b = np.array(point_b)
    return np.linalg.norm(a - b)

def calculate_EAR(eye_point):
    #Vertical distance 1: upper outer lid ->lower outer lid
    A = euclidean_distance(eye_point[1], eye_point[5])
    #Vertical distance 2: upper inner lid -> lower inner lid
    B = euclidean_distance(eye_point[2], eye_point[4])
    #Horizontal distance: left corner -> right corner
    C = euclidean_distance(eye_point[0], eye_point[3])
    #EAR formula from Soukupova & Cech 2016
    return (A + B) / (2.0 * C)

def get_eye_coords(landmarks, indices, w, h):
    coords =[]
    for idx in indices:
        lm = landmarks[idx]
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        coords.append((x_px, y_px))
    return coords

test_eye = [
    (0, 100),  #pts[0]  left corner
    (30, 85), #pts[1]  upper outer lid
    (60, 85), #pts[2]  upper inner lid
    (90, 100), #pts[3]  right corner
    (60, 115), #pts[4] lower inner lid
    (30, 115)  #pts[5] lower outer lid
]

ear = calculate_EAR(test_eye)
print(f"EAR: {ear:.4f}") #should be ~ 0.30
#------------------------------------------------------

#This Section is for MAR

MOUTH_IDX = [60, 40, 270, 0, 291, 375, 314, 146]
MOUTH_OUTLINE = [
        # OUTER UPPER LIP
    61, 185, 40, 39, 37, 0, 267,
    269, 270, 409, 291,

    # OUTER LOWER LIP
    375, 321, 405, 314, 17,
    84, 181, 91, 146,

    # INNER UPPER LIP
    78, 191, 80, 81, 82,
    13, 312, 311, 310, 415, 308,

    # INNER LOWER LIP
    324, 318, 402, 317, 14,
    87, 178, 88, 95
]

def calculate_MAR(mouth_points):
    #3 vertical distances: upper lip paired with lower lip
    A = euclidean_distance(mouth_points[1], mouth_points[7])
    B = euclidean_distance(mouth_points[2], mouth_points[6])
    C = euclidean_distance(mouth_points[2], mouth_points[5])
    #Horizontal distance: left corner to right corner
    D = euclidean_distance(mouth_points[0], mouth_points[4])
    #MAR Formula - 3 verticals / 3*horizontal
    return (A + B + C) / (3 * D)

test_mouth = [(0,50),
              (30,20),
              (70,20),
              (50,10),
              (100,50),
              (50,90),
              (30,80),
              (70,80)
              ]
print(f"MAR test:{calculate_MAR(test_mouth):.4f}")