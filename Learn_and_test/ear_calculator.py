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