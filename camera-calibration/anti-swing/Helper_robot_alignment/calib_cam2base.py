"""Determine CAM2BASE automatically.

The robot makes two small moves, along base +X then along base +Y, while a static
marker is watched. Because the camera is carried by the robot, a marker that is
fixed in the world appears to move in the opposite direction in the image; the
script takes care of the sign. It then prints the 2x2 matrix to paste into
frames.py.

Requirements: the marker must stay visible during both moves, the tool must not
rotate, and the crate must not move.
"""

import rtde_control
import rtde_receive
import numpy as np
import cv2
import pickle
import time

ROBOT_IP   = "192.168.56.102"
CAMERA_ID  = 0
ID_MARQUEUR = 8
TAILLE     = 0.157
CALIB_FILE = "output/calibration_data.pkl"
PAS        = 0.05                # size of each test move [m]
V, A       = 0.05, 0.2

with open(CALIB_FILE, "rb") as f:
    data = pickle.load(f)
mtx = np.array(data.get("camera_matrix", data.get("mtx")))
dist = np.array(data.get("distortion_coefficients", data.get("dist")))

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters())
h = TAILLE / 2
obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def position_marqueur(n=20):
    """Average position of the marker in the camera frame, over n frames."""
    vus = []
    while len(vus) < n:
        ok, frame = cap.read()
        if not ok:
            continue
        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None or ID_MARQUEUR not in ids.flatten():
            continue
        i = int(np.where(ids.flatten() == ID_MARQUEUR)[0][0])
        ok, rvec, tvec = cv2.solvePnP(obj, corners[i][0], mtx, dist)
        if ok:
            vus.append(tvec.flatten())
    return np.mean(vus, axis=0)


rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
pose0 = rtde_r.getActualTCPPose()

colonnes = []
for axe, nom in [(0, "+X"), (1, "+Y")]:
    print(f"move along base {nom} ...")
    p_avant = position_marqueur()
    cible = list(pose0)
    cible[axe] += PAS
    rtde_c.moveL(cible, V, A)
    time.sleep(0.5)
    p_apres = position_marqueur()
    rtde_c.moveL(list(pose0), V, A)
    time.sleep(0.5)

    # the camera moved by +PAS, so a static marker moved by -PAS in the image
    d = -(p_apres - p_avant)[:2] / PAS
    print(f"  observed direction in camera frame: {np.round(d, 3)}")
    colonnes.append(d)

M = np.column_stack(colonnes)            # camera -> base
M = np.round(M)                          # snap to the nearest axis permutation
print("\nCAM2BASE = np.array([[%.0f, %.0f],\n                     [%.0f, %.0f]])"
      % (M[0, 0], M[0, 1], M[1, 0], M[1, 1]))

cap.release()
rtde_c.stopScript()