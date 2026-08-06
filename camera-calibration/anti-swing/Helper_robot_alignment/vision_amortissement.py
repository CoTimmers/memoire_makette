"""Vision for the damping test: the load marker alone.

The full vision.py needs two markers, because it has to express a position
error in a world frame. The damping test does not: the tool returns to its own
starting point, and the only quantity measured is the sway angle. Requiring the
reference marker there would add a constraint for nothing, and a demanding one,
since both markers would have to stay in frame while the load swings.

So this module reads ID_CHARGE only.

Shared dict, updated by its own thread:

    theta       [thx, thy]   sway angles, robot base axes               [rad]
    vus         bool         load marker currently detected
    pret        bool         pivot calibrated, theta is meaningful
    t           float        instant the image was TAKEN (delay subtracted)
    l_mes       float        measured pivot-to-load distance, consistency check

theta_dot is not computed here: the Kalman filter reconstructs it, with far less
noise and with the camera delay compensated.

The pivot is located once, from the load hanging at rest: the cable is then
vertical, so the pivot sits l_cable above the attachment point. Everything
afterwards is measured from it, which is why the load must be still during
calibration - a swinging load gives a pivot averaged over the swing.
"""

import cv2
import numpy as np
import pickle
import threading
import time

# ---------------- configuration ----------------
CAMERA_ID  = 0
CALIB_FILE = "output/calibration_data.pkl"

ID_CHARGE     = 8               # crate marker (the load)
TAILLE_CHARGE = 0.157           # printed side [m]

OFFSET_ACCROCHE = np.array([-0.05, 0.12, 0.0])   # marker -> cable attachment

# Camera horizontal axes -> robot base axes. From calib_cam2base.py.
CAM2BASE = np.array([[1.0, 0.0],
                     [0.0, -1.0]])

RETARD_CAM = 0.031              # capture -> availability [s]
N_CALIB    = 60                 # frames used to locate the cable pivot

AFFICHAGE = False               # True to open the debug window

_stop = threading.Event()


def _obj_points(taille):
    h = taille / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                    dtype=np.float32)


def _pose(corners, ids, cible_id, taille, mtx, dist):
    if ids is None:
        return None
    idx = np.where(ids.flatten() == cible_id)[0]
    if len(idx) == 0:
        return None
    ok, rvec, tvec = cv2.solvePnP(_obj_points(taille), corners[idx[0]][0],
                                  mtx, dist)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return tvec.flatten(), R


def _projette(p, mtx, dist):
    pt, _ = cv2.projectPoints(np.asarray(p, dtype=np.float64).reshape(1, 3),
                              np.zeros(3), np.zeros(3), mtx, dist)
    return int(pt[0][0][0]), int(pt[0][0][1])


def _loop(etat, l_cable):
    with open(CALIB_FILE, "rb") as f:
        data = pickle.load(f)
    mtx = np.array(data.get("camera_matrix", data.get("mtx")))
    dist = np.array(data.get("distortion_coefficients", data.get("dist")))

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"camera {CAMERA_ID} impossible a ouvrir")

    pivot, calib = None, []

    while not _stop.is_set():
        ok, frame = cap.read()
        # cap.read() blocks until the image is available, so the timestamp is
        # taken here and the pipeline delay is subtracted once, not twice.
        t = time.perf_counter() - RETARD_CAM
        if not ok:
            continue

        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        charge = _pose(corners, ids, ID_CHARGE, TAILLE_CHARGE, mtx, dist)
        etat["vus"] = charge is not None

        if charge is None:
            if AFFICHAGE:
                cv2.putText(frame, "marqueur de charge non detecte", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow("vision amortissement", frame)
                cv2.waitKey(1)
            continue

        p_c, R_c = charge
        accroche = p_c + R_c @ OFFSET_ACCROCHE

        # ---- pivot calibration: load at rest, cable vertical ----
        if pivot is None:
            calib.append(accroche)
            if len(calib) >= N_CALIB:
                pivot = np.mean(calib, axis=0) - np.array([0.0, 0.0, l_cable])
                etat["pret"] = True
                print(f"\npivot calibre: {np.round(pivot, 4)}")
            if AFFICHAGE:
                cv2.putText(frame, f"calibration du pivot {len(calib)}/{N_CALIB}"
                            " - ne pas toucher la charge", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow("vision amortissement", frame)
                cv2.waitKey(1)
            continue

        # ---- sway angles ----
        c = accroche - pivot
        th = np.array([np.arctan2(c[0], c[2]), np.arctan2(c[1], c[2])])
        etat["theta"] = CAM2BASE @ th
        etat["l_mes"] = float(np.linalg.norm(c))
        etat["t"] = t

        # ---- debug window ----
        if AFFICHAGE:
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            u_a = _projette(accroche, mtx, dist)
            cv2.circle(frame, u_a, 6, (0, 165, 255), 2)
            cv2.putText(frame, "accroche", (u_a[0] + 14, u_a[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1,
                        cv2.LINE_AA)
            # the vertical through the pivot: theta is the angle to this line
            u_p = _projette(pivot, mtx, dist)
            cv2.line(frame, u_p, u_a, (255, 255, 0), 1, cv2.LINE_AA)

            lignes = [f"theta   {np.degrees(etat['theta'][0]):+6.2f}, "
                      f"{np.degrees(etat['theta'][1]):+6.2f} deg",
                      f"l_mes   {etat['l_mes']:.3f} m  (attendu {l_cable:.3f})"]
            for i, s in enumerate(lignes):
                cv2.putText(frame, s, (10, 32 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(frame, s, (10, 32 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
                            cv2.LINE_AA)
            cv2.imshow("vision amortissement", frame)
            cv2.waitKey(1)

    cap.release()
    if AFFICHAGE:
        cv2.destroyAllWindows()


def start(etat, l_cable):
    etat.update({"theta": np.zeros(2), "vus": False, "pret": False,
                 "t": time.perf_counter(), "l_mes": l_cable})
    _stop.clear()
    threading.Thread(target=_loop, args=(etat, l_cable), daemon=True).start()


def stop():
    _stop.set()


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    # Displays theta live. Push the load by hand and check that the sign and
    # the magnitude are what you expect before running a real test.
    AFFICHAGE = True
    etat = {}
    start(etat, 1.17)
    print("charge immobile pour la calibration du pivot...")
    try:
        while not etat["pret"]:
            time.sleep(0.1)
        print("pret. Ctrl-C pour arreter.\n")
        while True:
            print(f"\rtheta {np.degrees(etat['theta'][0]):+6.2f}, "
                  f"{np.degrees(etat['theta'][1]):+6.2f} deg   "
                  f"l_mes {etat['l_mes']:.3f} m   "
                  f"vu {etat['vus']}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\narret.")
    finally:
        stop()
        time.sleep(0.3)