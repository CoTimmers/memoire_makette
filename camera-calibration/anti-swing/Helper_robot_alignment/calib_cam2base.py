"""Determine CAM2BASE automatically."""

import rtde_control
import rtde_receive
import numpy as np
import cv2
import pickle
import time
from pathlib import Path

ROBOT_IP    = "192.168.56.102"
CAMERA_ID   = 0
ID_MARQUEUR = 8
TAILLE      = 0.157
CALIB_FILE  = "output/calibration_data.pkl"
PAS         = 0.05
V, A        = 0.05, 0.2

with open(CALIB_FILE, "rb") as f:
    data = pickle.load(f)
mtx  = np.array(data.get("camera_matrix", data.get("mtx")))
dist = np.array(data.get("distortion_coefficients", data.get("dist")))

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters())
h = TAILLE / 2
obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def position_marqueur(n=20, timeout=10.0):
    for _ in range(5):          # vide le buffer accumulé pendant le moveL
        cap.read()
    vus = []
    t0 = time.time()
    nb_frames = nb_read_ko = nb_sans_marqueur = 0
    while len(vus) < n:
        if time.time() - t0 > timeout:
            print(f"  DIAG: frames={nb_frames} read_ko={nb_read_ko} "
                  f"sans_marqueur={nb_sans_marqueur} detections={len(vus)}")
            raise SystemExit("timeout detection")
        ok, frame = cap.read()
        nb_frames += 1
        if not ok:
            nb_read_ko += 1
            continue
        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None or ID_MARQUEUR not in ids.flatten():
            nb_sans_marqueur += 1
            continue
        i = int(np.where(ids.flatten() == ID_MARQUEUR)[0][0])
        ok, rvec, tvec = cv2.solvePnP(obj, corners[i][0], mtx, dist)
        if ok:
            vus.append(tvec.flatten())
    return np.mean(vus, axis=0)


rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
RC = rtde_control.RTDEControlInterface
rtde_c = RC(ROBOT_IP, 125.0, RC.FLAG_UPLOAD_SCRIPT | RC.FLAG_UPPER_RANGE_REGISTERS)

pose0 = rtde_r.getActualTCPPose()
print("pose de depart:", [round(x, 3) for x in pose0])
input("Entree pour lancer : ")

colonnes = []
try:
    for axe, nom in [(0, "+X"), (1, "+Y")]:
        print(f"move along base {nom} ...")
        p_avant = position_marqueur()

        cible = list(pose0)
        cible[axe] += PAS
        rtde_c.moveL(cible, V, A)
        time.sleep(0.5)
        if abs(rtde_r.getActualTCPPose()[axe] - cible[axe]) > 0.005:
            raise SystemExit(f"mouvement {nom} non effectue")

        p_apres = position_marqueur()

        rtde_c.moveL(list(pose0), V, A)
        time.sleep(0.5)

        d = -(p_apres - p_avant)[:2] / PAS
        print(f"  observed direction in camera frame: {np.round(d, 3)}")
        colonnes.append(d)

    M = np.column_stack(colonnes)
    print("\nbrut:\n", np.round(M, 3))
    Mr = np.round(M)
    print("det =", np.linalg.det(Mr))
    print("\nCAM2BASE = np.array([[%.0f, %.0f],\n                     [%.0f, %.0f]])"
          % (Mr[0, 0], Mr[0, 1], Mr[1, 0], Mr[1, 1]))

finally:
    try:
        rtde_c.moveL(list(pose0), V, A)   # retour garanti meme sur Ctrl+C
    except Exception:
        pass
    cap.release()
    rtde_c.stopScript()