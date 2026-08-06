"""Vision: two ArUco markers turned into the numbers the controller needs.

    charge      marker on the crate, which hangs from the crane cable
    reference   marker on the helper robot, static, defines the world frame

Shared dict, updated by its own thread:

    erreur      [ex, ey]     from the attachment point to the centre of the
                             target zone, in robot base axes              [m]
    theta       [thx, thy]   raw sway angles, base axes                 [rad]
    yaw         float        rotation of the crate about the vertical, relative
                             to the world frame                         [rad]
    err_monde   [ex, ey]     same error, along the world axes, for the zone test
    dans_cible  bool         attachment point inside the target rectangle
    vus         (ref, charge)
    pret        bool         pivot calibrated and an error already computed
    t           float        instant the image was TAKEN (delay subtracted)
    l_mes       float        measured pivot-to-load distance, consistency check

theta_dot is deliberately not computed here: the Kalman filter reconstructs it,
with far less noise and with the camera delay compensated.

yaw is measured but not controlled: the tool never rotates, and a cable
transmits no torque about its own axis, so nothing in the loop can act on it.
It is logged because the crate has to end up aligned with the target zone, and
because a yaw that grows during transport signals the load rotating freely.
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
ID_REF        = 12              # helper marker (static world reference)
TAILLE_CHARGE = 0.157           # printed side [m]
TAILLE_REF    = 0.100           # printed side [m]

# Offsets in each marker's own frame, so they rotate with it.
OFFSET_ORIGINE  = np.array([-0.07, 0.095, 0.0])   # ref marker    -> world origin
OFFSET_ACCROCHE = np.array([-0.05, 0.12, 0.0])   # crate marker  -> cable attachment
CRANE_TARGET    = np.array([-0.15, -0.17, 0.0])  # world origin  -> centre of the zone
CIBLE_DEMI      = np.array([0.030, 0.030])       # half-size of the zone [m]
OFFSET_LONG_SIDE_ALIGNED  = np.array([-0.155, -0.225, 0.0])
OFFSET_SHORT_SIDE_ALIGNED = np.array([ -0.25, -0.16, 0.0])

# Camera horizontal axes -> robot base axes. From calib_cam2base.py.
CAM2BASE = np.array([[1.0, 0.0],
                     [0.0, -1.0]])

RETARD_CAM = 0.031              # capture -> availability [s], from mesure_retard.py
N_CALIB    = 60                 # frames used to locate the cable pivot

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

coin_f = None
R_r_f = None
ALPHA_REF = 0.5  

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

    pivot, calib = None, []
    coin_f, R_ref_f = None, None        # repère monde lissé

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
        accroche = p_c + R_c @ OFFSET_ACCROCHE

        # ---- pivot calibration: load at rest, cable vertical ----
        if pivot is None:
            calib.append(accroche)
            if len(calib) >= N_CALIB:
                pivot = np.mean(calib, axis=0) - np.array([0.0, 0.0, l_cable])
                print(f"\npivot calibre: {np.round(pivot, 4)}")
            continue

        # ---- raw sway angles ----
        c = accroche - pivot
        th = np.array([np.arctan2(c[0], c[2]), np.arctan2(c[1], c[2])])
        etat["theta"] = CAM2BASE @ th
        etat["l_mes"] = float(np.linalg.norm(c))

        # ---- world frame, low-pass filtered ----
        # The reference marker is static, so heavy smoothing costs nothing and
        # removes the orientation jitter that the target squares amplify.
        if ref is not None:
            p_r, R_r = ref
            coin_brut = p_r + R_r @ OFFSET_ORIGINE
            if coin_f is None:
                coin_f, R_ref_f = coin_brut, R_r
            else:
                coin_f = (1 - ALPHA_REF) * coin_f + ALPHA_REF * coin_brut
                R_ref_f = (1 - ALPHA_REF) * R_ref_f + ALPHA_REF * R_r
                U, _, Vt = np.linalg.svd(R_ref_f)   # back onto SO(3)
                R_ref_f = U @ Vt

        # ---- position error and drop zones ----
        coin = cible = zones = None
        dans, zone_ok = False, None
        if coin_f is not None:
            coin, R_ref = coin_f, R_ref_f
            cible = coin + R_ref @ CRANE_TARGET

            # command error: attachment point -> point the crane aims at
            etat["erreur"] = CAM2BASE @ (cible - accroche)[:2]

            zones = {"long":  coin + R_ref @ OFFSET_LONG_SIDE_ALIGNED,
                     "short": coin + R_ref @ OFFSET_SHORT_SIDE_ALIGNED}

            err_monde = None
            for nom, centre in zones.items():
                e = (R_ref.T @ (accroche - centre))[:2]
                if err_monde is None or np.linalg.norm(e) < np.linalg.norm(err_monde):
                    err_monde = e                  # distance to the nearest zone
                if np.all(np.abs(e) < CIBLE_DEMI):
                    dans, zone_ok = True, nom

            etat["err_monde"] = err_monde
            etat["dans_cible"] = dans
            etat["zone"] = zone_ok

            # Yaw: angle of the crate x axis, projected on the world plane.
            x_c = R_ref.T @ R_c[:, 0]
            etat["yaw"] = float(np.arctan2(x_c[1], x_c[0]))

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

            if zones is not None:
                u_c = _triede(frame, coin, R_ref, mtx, dist)
                cv2.circle(frame, u_c, 6, (255, 255, 0), 2)
                cv2.putText(frame, "origine monde", (u_c[0] + 14, u_c[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                            cv2.LINE_AA)

                w, h = CIBLE_DEMI
                for nom, centre in zones.items():
                    sommets = [centre + R_ref @ np.array([sx * w, sy * h, 0.0])
                               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                    pts = np.array([_projette(p, mtx, dist) for p in sommets],
                                   np.int32)
                    couleur = (0, 200, 0) if zone_ok == nom else (0, 0, 255)
                    cv2.polylines(frame, [pts], True, couleur, 2, cv2.LINE_AA)
                    cv2.putText(frame, nom, (pts[3][0], pts[3][1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2,
                                cv2.LINE_AA)

                u_t = _projette(cible, mtx, dist)
                cv2.drawMarker(frame, u_t, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
                cv2.arrowedLine(frame, u_a, u_t, (0, 255, 255), 1, cv2.LINE_AA,
                                tipLength=0.05)

            e = etat.get("erreur", np.zeros(2))
            em = etat.get("err_monde", np.zeros(2))
            lignes = [f"error base   {1000*e[0]:+6.0f}, {1000*e[1]:+6.0f} mm",
                      f"error world  {1000*em[0]:+6.0f}, {1000*em[1]:+6.0f} mm  "
                      f"{'IN (' + str(zone_ok) + ')' if dans else 'out'}",
                      f"theta        {np.degrees(etat['theta'][0]):+5.1f}, "
                      f"{np.degrees(etat['theta'][1]):+5.1f} deg",
                      f"yaw          {np.degrees(etat['yaw']):+6.1f} deg",
                      f"l_mes        {etat['l_mes']:.3f} m  (expected "
                      f"{l_cable:.3f})",
                      f"seen         ref {ref is not None}   load True"]
            for i, s in enumerate(lignes):
                cv2.putText(frame, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("vision", frame)
            cv2.waitKey(1)

    cap.release()
    if AFFICHAGE:
        cv2.destroyAllWindows()


def start(etat, l_cable):
    etat.update({"theta": np.zeros(2), "err_monde": np.zeros(2), "yaw": 0.0,
                 "vus": (False, False), "dans_cible": False, "pret": False,
                 "t": time.perf_counter(), "l_mes": l_cable})
    threading.Thread(target=_loop, args=(etat, l_cable), daemon=True).start()


def stop():
    _stop.set()