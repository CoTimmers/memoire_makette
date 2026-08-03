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

AFFICHAGE = False               # True to open a debug window (validation only)

_stop = threading.Event()
_cible = np.zeros(2)            # (d1x, d1y) in the world frame
_lock = threading.Lock()


def set_target(d1x, d1y):
    """Change the target, expressed in the crate frame, while running."""
    global _cible
    with _lock:
        _cible = np.array([float(d1x), float(d1y)])


def _projette(p, mtx, dist):
    """Project a 3D point of the camera frame onto the image."""
    pt, _ = cv2.projectPoints(np.asarray(p).reshape(1, 3),
                              np.zeros(3), np.zeros(3), mtx, dist)
    return int(pt[0][0][0]), int(pt[0][0][1])


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
        coin = cible = None
        if bac is not None:
            p_b, R_b = bac
            coin = p_b + R_b @ OFFSET_COIN               # origin of the world frame
            with _lock:
                d = _cible.copy()
            cible = coin + R_b @ np.array([d[0], d[1], 0.0])
            etat["erreur"] = CAM2BASE @ (accroche - cible)[:2]

        etat["t"] = t
        etat["pret"] = "erreur" in etat

        # ---- debug window ----
        if AFFICHAGE:
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            u_a = _projette(accroche, mtx, dist)
            cv2.circle(frame, u_a, 8, (0, 165, 255), 2)
            cv2.putText(frame, "helper", (u_a[0] + 12, u_a[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
            if coin is not None:
                u_c = _projette(coin, mtx, dist)
                u_t = _projette(cible, mtx, dist)
                cv2.circle(frame, u_c, 6, (255, 0, 0), 2)
                cv2.putText(frame, "corner = world origin", (u_c[0] + 12, u_c[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.drawMarker(frame, u_t, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                cv2.putText(frame, "target", (u_t[0] + 12, u_t[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.arrowedLine(frame, u_a, u_t, (0, 0, 255), 2, tipLength=0.06)
            e = etat.get("erreur", np.zeros(2))
            lignes = [f"error  {1000*e[0]:+6.0f}, {1000*e[1]:+6.0f} mm",
                      f"theta  {np.degrees(etat['theta'][0]):+5.1f}, "
                      f"{np.degrees(etat['theta'][1]):+5.1f} deg",
                      f"l_mes  {etat['l_mes']:.3f} m",
                      f"seen   crate {bac is not None}   helper True"]
            for i, s in enumerate(lignes):
                cv2.putText(frame, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("frames debug", frame)
            cv2.waitKey(1)

    cap.release()
    if AFFICHAGE:
        cv2.destroyAllWindows()


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