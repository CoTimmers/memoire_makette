"""Vision module: turns two ArUco markers into the numbers the controller needs.

Two markers are tracked:
  - the crate, which is static on the pile and defines the world frame:
    origin at its corner, axes along its edges;
  - the helper robot, which hangs from the crane cable and is the load being moved.

From them this module produces, in a shared dict updated by its own thread:

    erreur     [ex, ey]        position error of the helper w.r.t. the target,
                               in robot base axes                          [m]
    theta      [th_x, th_y]    sway angles of the cable                    [rad]
    theta_dot  [thd_x, thd_y]  their time derivatives                    [rad/s]
    vus        (bac, helper)   whether each marker is currently seen
    pret       bool            pivot calibrated, measurements valid
    t          float           timestamp of the last valid measurement

Everything geometric lives here. The controller only sees these numbers.

    import frames
    etat = {}
    frames.start(etat, L_CABLE, d1=(0.30, 0.10))
    while not etat["pret"]:
        time.sleep(0.1)
    ...
    frames.set_target(0.10, -0.05)      # next waypoint, same run
    ...
    frames.stop()
"""

import cv2
import numpy as np
import pickle
import threading
import time

# ---------------- configuration ----------------
CAMERA_ID   = 0
CALIB_FILE  = "output/calibration_data.pkl"

ID_BAC      = 8                 # marker on the crate  (world reference)
ID_HELPER   = 12                # marker on the helper robot (load)
TAILLE_BAC    = 0.157           # printed side of the crate marker  [m]
TAILLE_HELPER = 0.100           # printed side of the helper marker [m]

# Offsets, expressed in each marker's own frame, so they rotate with it.
OFFSET_COIN   = np.array([0.0, 0.13, 0.0])   # crate marker  -> corner of the crate
OFFSET_ACCROCHE = np.array([0.0, 0.0, 0.0])  # helper marker -> cable attachment point

# Camera horizontal axes -> robot base axes. Identity assumes cam x // base x.
# [[-1,0],[0,1]] inverts x, [[0,1],[1,0]] swaps the axes, etc.
CAM2BASE = np.array([[1.0, 0.0],
                     [0.0, 1.0]])

N_CALIB = 60                    # frames used to locate the cable pivot
N_DERIV = 3                     # frames used to differentiate theta

_stop = threading.Event()
_cible = np.zeros(2)            # (d1x, d1y) in the world frame
_lock = threading.Lock()


def set_target(d1x, d1y):
    """Change the target, expressed in the crate frame, while running."""
    global _cible
    with _lock:
        _cible = np.array([float(d1x), float(d1y)])


def _obj_points(taille):
    h = taille / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


def _pose(corners, ids, cible_id, taille, mtx, dist):
    """Return (position, rotation) of one marker in the camera frame, or None."""
    if ids is None:
        return None
    idx = np.where(ids.flatten() == cible_id)[0]
    if len(idx) == 0:
        return None
    ok, rvec, tvec = cv2.solvePnP(_obj_points(taille), corners[idx[0]][0], mtx, dist)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return tvec.flatten(), R


def _loop(etat, l_cable):
    with open(CALIB_FILE, "rb") as f:
        data = pickle.load(f)
    mtx = np.array(data.get("camera_matrix", data.get("mtx")))
    dist = np.array(data.get("distortion_coefficients", data.get("dist")))

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())

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
        bac = _pose(corners, ids, ID_BAC, TAILLE_BAC, mtx, dist)
        helper = _pose(corners, ids, ID_HELPER, TAILLE_HELPER, mtx, dist)
        etat["vus"] = (bac is not None, helper is not None)

        if helper is None:
            continue
        p_h, R_h = helper
        accroche = p_h + R_h @ OFFSET_ACCROCHE          # cable attachment point

        # ---- pivot calibration: helper at rest, cable vertical ----
        if pivot is None:
            calib.append(accroche)
            if len(calib) >= N_CALIB:
                pivot = np.mean(calib, axis=0) - np.array([0.0, 0.0, l_cable])
                print(f"\npivot calibrated: {np.round(pivot, 4)}")
            continue

        # ---- sway, from the cable vector ----
        c = accroche - pivot
        th = np.array([np.arctan2(c[0], c[2]), np.arctan2(c[1], c[2])])
        histo.append((t, th))
        if len(histo) > N_DERIV + 1:
            histo.pop(0)
        thd = ((th - histo[0][1]) / (t - histo[0][0])
               if len(histo) == N_DERIV + 1 else np.zeros(2))

        etat["theta"] = CAM2BASE @ th
        etat["theta_dot"] = CAM2BASE @ thd
        etat["l_mes"] = float(np.linalg.norm(c))

        # ---- position error, only when the crate is visible ----
        if bac is not None:
            p_b, R_b = bac
            coin = p_b + R_b @ OFFSET_COIN               # origin of the world frame
            with _lock:
                d = _cible.copy()
            cible = coin + R_b @ np.array([d[0], d[1], 0.0])
            etat["erreur"] = CAM2BASE @ (accroche - cible)[:2]

        etat["t"] = t
        etat["pret"] = "erreur" in etat

    cap.release()


def start(etat, l_cable, d1=(0.0, 0.0)):
    set_target(*d1)
    for k, v in [("erreur", np.zeros(2)), ("theta", np.zeros(2)),
                 ("theta_dot", np.zeros(2)), ("vus", (False, False)),
                 ("pret", False), ("t", time.perf_counter()),
                 ("l_mes", l_cable)]:
        etat.setdefault(k, v)
    etat.pop("erreur", None)                 # so that 'pret' only turns True
    threading.Thread(target=_loop, args=(etat, l_cable), daemon=True).start()


def stop():
    _stop.set()