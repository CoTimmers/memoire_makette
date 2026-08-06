"""Vision for experiment 7: swing the load into the crate, then damp it out.

The two roles are swapped with respect to experiment 6, because the object that
hangs from the cable is no longer the same:

    ID_CHARGE = 12   marker on the hanging load, the one that swings
    ID_REF    =  8   marker on the crate, static, defines the world frame

Neither marker was moved on its object, so each keeps the printed size and the
offset that were measured for it. Marker 12 measured 100 mm and its offset was
[-0.07, 0.095, 0]; both values are simply reused under their new names.

Two fixed points are defined in the world frame:

    D3   near the origin, where the crane stops dead so that the load carries on
         and reaches the crate
    D1   far from the origin, the retreat point where the sway is damped out

Both lie on the same 45 degree ray, so the whole experiment happens along one
diagonal and the two horizontal axes are excited equally.

Shared dict, updated by its own thread:

    err_d1      [ex, ey]     load pivot -> D1, robot base axes            [m]
    err_d3      [ex, ey]     load pivot -> D3, robot base axes            [m]
    erreur      [ex, ey]     whichever of the two CIBLE_ACTIVE names      [m]
    pos_monde   [x, y]       load pivot in world axes, origin at the crate [m]
    dist_origine float       norm of pos_monde, distance to the crate      [m]
    theta       [thx, thy]   raw sway angles, base axes                  [rad]
    yaw         float        load rotation about the vertical            [rad]
    vus         (ref, charge)
    pret        bool         suspension calibrated and an error computed
    t           float        instant the image was TAKEN (delay subtracted)
    l_mes       float        measured suspension-to-load distance

Two different points are called a pivot in this experiment, so the naming is
kept strict:

    suspension     where the cable meets the tool. Fixed in the camera frame,
                   found once by calibration while the load hangs still.
    pivot_charge   where the cable meets the load, OFFSET_PIVOT from marker 12.
                   This is the point the controller drives, and the point whose
                   distance to D1 and D3 is the error.
"""

import cv2
import numpy as np
import pickle
import threading
import time

# ---------------- configuration ----------------
CAMERA_ID  = 0
CALIB_FILE = "output/calibration_data.pkl"

ID_CHARGE     = 12              # hanging load
ID_REF        = 8               # crate, static world reference
TAILLE_CHARGE = 0.100           # printed side of marker 12 [m]
TAILLE_REF    = 0.157           # printed side of marker 8  [m]

# Offsets in each marker's own frame, so they rotate with it.
OFFSET_PIVOT   = np.array([-0.07, 0.095, 0.0])   # marker 12 -> cable attachment
OFFSET_ORIGINE = np.array([ 0.00, 0.000, 0.0])   # marker 8  -> world origin
                                                 # MEASURE THIS ON THE BENCH

# The two working points, in world axes. Both on the 45 degree diagonal:
# |D1| = 400 mm, |D3| = 50 mm.
D1 = np.array([0.2828, 0.2828, 0.0])
D3 = np.array([0.0353, 0.0353, 0.0])

RAYON_OK = 0.030                # radius within which D1 counts as reached [m]

# Which point the "erreur" key follows. 7_main.py writes this.
CIBLE_ACTIVE = "d3"

# Camera horizontal axes -> robot base axes. From calib_cam2base.py.
CAM2BASE = np.array([[1.0, 0.0],
                     [0.0, -1.0]])

RETARD_CAM = 0.031              # capture -> availability [s], from mesure_retard.py
N_CALIB    = 60                 # frames used to locate the cable suspension
ALPHA_REF  = 0.5                # low-pass on the world frame

AFFICHAGE = False               # True to open the debug window

_stop = threading.Event()


def _projette(p, mtx, dist):
    pt, _ = cv2.projectPoints(np.asarray(p, dtype=np.float64).reshape(1, 3),
                              np.zeros(3), np.zeros(3), mtx, dist)
    return int(pt[0][0][0]), int(pt[0][0][1])


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
        cv2.putText(frame, nom, p, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    couleur, 1, cv2.LINE_AA)
    return o


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

    suspension, calib = None, []
    o_f, R_ref_f = None, None           # smoothed world frame

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
        ref = _pose(corners, ids, ID_REF, TAILLE_REF, mtx, dist)
        etat["vus"] = (ref is not None, charge is not None)

        if charge is None:
            continue
        p_c, R_c = charge
        pivot_charge = p_c + R_c @ OFFSET_PIVOT

        # ---- suspension calibration: load at rest, cable vertical ----
        if suspension is None:
            calib.append(pivot_charge)
            if len(calib) >= N_CALIB:
                suspension = np.mean(calib, axis=0) - np.array([0.0, 0.0, l_cable])
                print(f"\nsuspension calibree: {np.round(suspension, 4)}")
            continue

        # ---- raw sway angles ----
        c = pivot_charge - suspension
        th = np.array([np.arctan2(c[0], c[2]), np.arctan2(c[1], c[2])])
        etat["theta"] = CAM2BASE @ th
        etat["l_mes"] = float(np.linalg.norm(c))

        # ---- world frame, low-pass filtered ----
        # The crate is static, so heavy smoothing costs nothing and removes the
        # orientation jitter that a 40 cm lever arm would amplify at D1.
        if ref is not None:
            p_r, R_r = ref
            o_brut = p_r + R_r @ OFFSET_ORIGINE
            if o_f is None:
                o_f, R_ref_f = o_brut, R_r
            else:
                o_f = (1 - ALPHA_REF) * o_f + ALPHA_REF * o_brut
                R_ref_f = (1 - ALPHA_REF) * R_ref_f + ALPHA_REF * R_r
                U, _, Vt = np.linalg.svd(R_ref_f)       # back onto SO(3)
                R_ref_f = U @ Vt

        # ---- errors to the two working points ----
        p_d1 = p_d3 = None
        if o_f is not None:
            o, R_ref = o_f, R_ref_f
            p_d1 = o + R_ref @ D1
            p_d3 = o + R_ref @ D3

            etat["err_d1"] = CAM2BASE @ (p_d1 - pivot_charge)[:2]
            etat["err_d3"] = CAM2BASE @ (p_d3 - pivot_charge)[:2]
            etat["erreur"] = (etat["err_d1"] if CIBLE_ACTIVE == "d1"
                              else etat["err_d3"])

            # where the load is along the diagonal, crate at the origin
            pos = (R_ref.T @ (pivot_charge - o))[:2]
            etat["pos_monde"] = pos
            etat["dist_origine"] = float(np.linalg.norm(pos))
            etat["a_d1"] = bool(np.linalg.norm(etat["err_d1"]) < RAYON_OK)

            # Yaw: angle of the load x axis, projected on the world plane.
            x_c = R_ref.T @ R_c[:, 0]
            etat["yaw"] = float(np.arctan2(x_c[1], x_c[0]))

        etat["t"] = t
        etat["pret"] = "err_d3" in etat

        # ---- debug window ----
        if AFFICHAGE:
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            u_p = _triede(frame, pivot_charge, R_c, mtx, dist)
            cv2.circle(frame, u_p, 6, (0, 165, 255), 2)
            cv2.putText(frame, "pivot charge", (u_p[0] + 14, u_p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

            if p_d1 is not None:
                u_o = _triede(frame, o, R_ref, mtx, dist)
                cv2.circle(frame, u_o, 6, (255, 255, 0), 2)
                cv2.putText(frame, "origine (bac)", (u_o[0] + 14, u_o[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                            cv2.LINE_AA)

                for nom, p in (("D3", p_d3), ("D1", p_d1)):
                    u = _projette(p, mtx, dist)
                    actif = (nom.lower() == CIBLE_ACTIVE)
                    couleur = (0, 200, 0) if actif else (200, 200, 200)
                    cv2.drawMarker(frame, u, couleur, cv2.MARKER_TILTED_CROSS,
                                   16, 2 if actif else 1)
                    cv2.circle(frame, u, 5, couleur, 1)
                    cv2.putText(frame, nom, (u[0] + 10, u[1] + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2,
                                cv2.LINE_AA)

                u_act = _projette(p_d1 if CIBLE_ACTIVE == "d1" else p_d3,
                                  mtx, dist)
                cv2.arrowedLine(frame, u_p, u_act, (0, 255, 255), 1, cv2.LINE_AA,
                                tipLength=0.05)
                cv2.line(frame, _projette(o, mtx, dist),
                         _projette(p_d1, mtx, dist), (120, 120, 120), 1,
                         cv2.LINE_AA)

            e = etat.get("erreur", np.zeros(2))
            pos = etat.get("pos_monde", np.zeros(2))
            lignes = [f"cible active {CIBLE_ACTIVE.upper()}   "
                      f"error {1000*e[0]:+6.0f}, {1000*e[1]:+6.0f} mm  "
                      f"|e| {1000*np.linalg.norm(e):5.0f} mm",
                      f"charge/monde {1000*pos[0]:+6.0f}, {1000*pos[1]:+6.0f} mm  "
                      f"dist origine {1000*etat.get('dist_origine', 0):5.0f} mm",
                      f"theta        {np.degrees(etat['theta'][0]):+5.1f}, "
                      f"{np.degrees(etat['theta'][1]):+5.1f} deg",
                      f"yaw          {np.degrees(etat.get('yaw', 0.0)):+6.1f} deg",
                      f"l_mes        {etat['l_mes']:.3f} m  (expected "
                      f"{l_cable:.3f})",
                      f"seen         ref(8) {ref is not None}   "
                      f"charge(12) True"]
            for i, s in enumerate(lignes):
                cv2.putText(frame, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("vision 7", frame)
            cv2.waitKey(1)

    cap.release()
    if AFFICHAGE:
        cv2.destroyAllWindows()


def start(etat, l_cable):
    etat.update({"theta": np.zeros(2), "pos_monde": np.zeros(2), "yaw": 0.0,
                 "dist_origine": 0.0, "a_d1": False,
                 "vus": (False, False), "pret": False,
                 "t": time.perf_counter(), "l_mes": l_cable})
    threading.Thread(target=_loop, args=(etat, l_cable), daemon=True).start()


def stop():
    _stop.set()