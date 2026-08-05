"""Mesure l'angle de balancement du bac. Un seul marqueur."""

import cv2
import numpy as np
import pickle
import threading
import time

CAMERA_ID   = 0
CALIB_FILE  = "output/calibration_data.pkl"

ID_CHARGE     = 8
TAILLE_CHARGE = 0.157
OFFSET_ACCROCHE = np.array([-0.05, 0.11, 0.0])

CAM2BASE = np.array([[1.0,  0.0],
                     [0.0, -1.0]])

RETARD_CAM = 0.045              # retard capture -> traitement [s]
N_CALIB    = 60                 # frames pour localiser le pivot

_stop = threading.Event()


def _obj_points(taille):
    h = taille / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                    dtype=np.float32)


def _loop(etat, l_cable):
    with open(CALIB_FILE, "rb") as f:
        data = pickle.load(f)
    mtx  = np.array(data.get("camera_matrix", data.get("mtx")))
    dist = np.array(data.get("distortion_coefficients", data.get("dist")))

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    obj = _obj_points(TAILLE_CHARGE)
    pivot, calib = None, []

    while not _stop.is_set():
        t = time.perf_counter() - RETARD_CAM
        ok, frame = cap.read()
        if not ok:
            continue

        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        etat["vu"] = ids is not None and ID_CHARGE in ids.flatten()
        if not etat["vu"]:
            continue

        i = int(np.where(ids.flatten() == ID_CHARGE)[0][0])
        ok, rvec, tvec = cv2.solvePnP(obj, corners[i][0], mtx, dist)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        accroche = tvec.flatten() + R @ OFFSET_ACCROCHE

        if pivot is None:
            calib.append(accroche)
            if len(calib) >= N_CALIB:
                pivot = np.mean(calib, axis=0) - np.array([0.0, 0.0, l_cable])
                print(f"\npivot calibre: {np.round(pivot, 4)}")
                etat["pret"] = True
            continue

        c = accroche - pivot
        th = np.array([np.arctan2(c[0], c[2]), np.arctan2(c[1], c[2])])

        etat["theta"] = CAM2BASE @ th
        etat["l_mes"] = float(np.linalg.norm(c))
        etat["t"] = t

    cap.release()


def start(etat, l_cable):
    etat.update({"theta": np.zeros(2), "vu": False, "pret": False,
                 "t": time.perf_counter(), "l_mes": l_cable})
    threading.Thread(target=_loop, args=(etat, l_cable), daemon=True).start()


def stop():
    _stop.set()