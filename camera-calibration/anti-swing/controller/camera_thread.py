"""Camera thread: fills a shared dict with theta, theta_dot, t and pret.

    import camera_thread
    etat = {}
    camera_thread.start(etat, L_CABLE)
    while etat.get("pret") is not True:
        time.sleep(0.1)
    ...
    camera_thread.stop()
"""

import cv2
import numpy as np
import pickle
import threading
import time
import os

CAMERA_ID   = 0
MARKER_SIZE = 0.157
COM_OFFSET  = np.array([0.0, 0.13, 0.0])
CALIB_FILE  = "output/calibration_data.pkl"
N_CALIB     = 60
N_DERIV     = 3
SIGN        = +1                 # flip to -1 if the sign check fails

_stop = threading.Event()


def _loop(etat, l_cable):
    with open(CALIB_FILE, "rb") as f:
        data = pickle.load(f)
    mtx = np.array(data.get("camera_matrix", data.get("mtx")))
    dist = np.array(data.get("distortion_coefficients", data.get("dist")))

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())
    h = MARKER_SIZE / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    pivot, calib, histo = None, [], []
    while not _stop.is_set():
        ok, frame = cap.read()
        if not ok:
            continue
        t = time.perf_counter()
        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj, corners[0][0], mtx, dist)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        com = tvec.flatten() + R @ COM_OFFSET

        if pivot is None:
            calib.append(com)
            if len(calib) >= N_CALIB:
                pivot = np.mean(calib, axis=0) - np.array([0.0, 0.0, l_cable])
                print(f"\npivot calibrated: {np.round(pivot, 4)}")
            continue

        c = com - pivot
        theta = SIGN * float(np.arctan2(c[0], c[2]))
        histo.append((t, theta))
        if len(histo) > N_DERIV + 1:
            histo.pop(0)
        theta_dot = ((theta - histo[0][1]) / (t - histo[0][0])
                     if len(histo) == N_DERIV + 1 else 0.0)
        etat.update(theta=theta, theta_dot=theta_dot, t=t, pret=True)

    cap.release()


def start(etat, l_cable):
    etat.setdefault("theta", 0.0)
    etat.setdefault("theta_dot", 0.0)
    etat.setdefault("t", time.perf_counter())
    etat.setdefault("pret", False)
    threading.Thread(target=_loop, args=(etat, l_cable), daemon=True).start()


def stop():
    _stop.set()