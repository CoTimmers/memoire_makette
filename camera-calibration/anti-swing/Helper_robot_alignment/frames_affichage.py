"""Vision module: turns two ArUco markers into the numbers the controller needs.

Two markers are tracked:
  - the load (crate) hanging from the crane cable;
  - the helper robot, static on the floor, which defines the world frame.

Shared dict updated by its own thread:

    erreur     [ex, ey]        position error of the load w.r.t. the crane target,
                               in robot base axes                          [m]
    theta      [th_x, th_y]    sway angles of the cable                    [rad]
    theta_dot  [thd_x, thd_y]  their time derivatives                    [rad/s]
    vus        (ref, charge)   whether each marker is currently seen
    pret       bool            pivot calibrated, measurements valid
    t          float           timestamp of the last valid measurement
"""

import cv2
import numpy as np
import pickle
import threading
import time

# ---------------- configuration ----------------
CAMERA_ID   = 0
CALIB_FILE  = "output/calibration_data.pkl"

ID_CHARGE     = 8               # marker on the crate  (load, hangs from cable)
ID_REF        = 12              # marker on the helper (static world reference)
TAILLE_CHARGE = 0.157           # printed side of the crate marker  [m]
TAILLE_REF    = 0.100           # printed side of the helper marker [m]

# Offsets, expressed in each marker's own frame, so they rotate with it.
OFFSET_ORIGINE  = np.array([-0.05, 0.08, 0.0])   # helper marker -> world origin
OFFSET_ACCROCHE = np.array([-0.05, 0.11, 0.0])   # crate marker  -> cable attachment
CRANE_TARGET    = np.array([-0.12, -0.12, 0.0])  # world origin  -> crane target

# Camera horizontal axes -> robot base axes.
CAM2BASE = np.array([[1.0,  0.0],
                     [0.0, -1.0]])

N_CALIB = 60                    # frames used to locate the cable pivot
N_DERIV = 3                     # frames used to differentiate theta

AFFICHAGE = False               # True to open a debug window (validation only)

_stop = threading.Event()
_cible = np.zeros(2)
_lock = threading.Lock()


def set_target(d1x, d1y):
    """Kept for API compatibility; the target is now CRANE_TARGET."""
    global _cible
    with _lock:
        _cible = np.array([float(d1x), float(d1y)])


def _projette(p, mtx, dist):
    """Project a 3D point of the camera frame onto the image."""
    pt, _ = cv2.projectPoints(np.asarray(p, dtype=np.float64).reshape(1, 3),
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


def _triede(frame, origine, R, mtx, dist, longueur=0.06):
    """Draw a 3D frame: X red, Y green, Z blue. Returns the origin pixel."""
    o = _projette(origine, mtx, dist)
    for vec, couleur, nom in [((longueur, 0, 0), (0, 0, 255), "X"),
                              ((0, longueur, 0), (0, 255, 0), "Y"),
                              ((0, 0, longueur), (255, 0, 0), "Z")]:
        p = _projette(origine + R @ np.array(vec), mtx, dist)
        cv2.line(frame, o, p, couleur, 2, cv2.LINE_AA)
        cv2.putText(frame, nom, p, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, couleur, 1, cv2.LINE_AA)
    return o


def _loop(etat, l_cable):
    with open(CALIB_FILE, "rb") as f:
        data = pickle.load(f)
    mtx = np.array(data.get("camera_matrix", data.get("mtx")))
    dist = np.array(data.get("distortion_coefficients", data.get("dist")))

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    pivot, calib, histo = None, [], []

    while not _stop.is_set():
        ok, frame = cap.read()
        if not ok:
            continue
        t = time.perf_counter()

        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        charge = _pose(corners, ids, ID_CHARGE, TAILLE_CHARGE, mtx, dist)
        ref    = _pose(corners, ids, ID_REF,    TAILLE_REF,    mtx, dist)
        etat["vus"] = (ref is not None, charge is not None)

        if charge is None:
            continue
        p_c, R_c = charge
        accroche = p_c + R_c @ OFFSET_ACCROCHE          # cable attachment point

        # ---- pivot calibration: load at rest, cable vertical ----
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

        # ---- position error, only when the reference is visible ----
        coin = cible = None
        if ref is not None:
            p_r, R_r = ref
            coin = p_r + R_r @ OFFSET_ORIGINE            # world origin
            cible = coin + R_r @ CRANE_TARGET            # crane target
            etat["erreur"] = CAM2BASE @ (cible - accroche)[:2]

        etat["t"] = t
        etat["pret"] = "erreur" in etat

        # ---- debug window ----
        if AFFICHAGE:
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            u_a = _triede(frame, accroche, R_c, mtx, dist)
            cv2.circle(frame, u_a, 6, (0, 165, 255), 2)
            cv2.putText(frame, "accroche", (u_a[0] + 14, u_a[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

            if coin is not None:
                u_c = _triede(frame, coin, R_r, mtx, dist)
                cv2.circle(frame, u_c, 6, (255, 255, 0), 2)
                cv2.putText(frame, "origine monde", (u_c[0] + 14, u_c[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

                u_t = _projette(cible, mtx, dist)
                cv2.drawMarker(frame, u_t, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.circle(frame, u_t, 10, (0, 0, 255), 1)
                cv2.putText(frame, "cible grue", (u_t[0] + 14, u_t[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            e = etat.get("erreur", np.zeros(2))
            lignes = [f"error  {1000*e[0]:+6.0f}, {1000*e[1]:+6.0f} mm",
                      f"theta  {np.degrees(etat['theta'][0]):+5.1f}, "
                      f"{np.degrees(etat['theta'][1]):+5.1f} deg",
                      f"l_mes  {etat['l_mes']:.3f} m",
                      f"seen   ref {ref is not None}   charge True"]
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
    etat.pop("erreur", None)
    threading.Thread(target=_loop, args=(etat, l_cable), daemon=True).start()


def stop():
    _stop.set()