"""Vision for experiment 8: the crate is carried into a helper standing on the
floor, and the question is how much the helper moves.

Three markers:

    ID_CHARGE = 8    on the crate, which hangs from the cable
    ID_HELPER = 12   on the helper, standing on the floor, free to be pushed
    ID_REF    = 5    new marker on the floor, static

Why a third marker, when every offset was already measured against marker 12.
Two reasons, and neither is about the offsets themselves.

    An object cannot measure its own displacement. Referred to a frame attached
    to the helper, the helper never moves: the reading would be zero whatever
    happens. A fixed anchor is needed, and that is ID_REF.

    A live anchor would drag the target along. If the drop zones followed marker
    12 frame by frame, pushing the helper 30 mm would move the zones 30 mm too,
    and the crane would chase a target that runs away from it.

So the geometry is anchored to the helper pose FROZEN at startup, expressed in
the ID_REF frame. Every offset keeps the value measured against marker 12:

    OFFSET_ORIGINE, CRANE_TARGET, OFFSET_LONG_SIDE_ALIGNED,
    OFFSET_SHORT_SIDE_ALIGNED

are unchanged, they are just applied to the frozen pose instead of the live one.
Once frozen, the helper may disappear behind the crate without the zones moving;
only ID_REF has to stay visible.

Shared dict, updated by its own thread:

    erreur       [ex, ey]   cable attachment -> point the crane aims at, base
                            axes                                           [m]
    theta        [thx, thy] raw sway angles, base axes                   [rad]
    err_monde    [ex, ey]   distance to the nearest drop zone, anchor axes [m]
    dans_cible   bool       attachment point inside a drop zone
    zone         str        "long", "short" or None
    yaw          float      crate rotation about the vertical             [rad]

    helper_pos   [x, y]     helper in the ID_REF frame                     [m]
    helper_depl  [dx, dy]   displacement since the frozen pose, anchor axes [m]
    helper_dyaw  float      rotation since the frozen pose                [rad]
    helper_bouge bool       displacement past SEUIL_BOUGE
    helper_pret  bool       frozen pose captured

    vus          (ref, charge, helper)
    pret         bool       everything calibrated and an error available
    t            float      instant the image was TAKEN (delay subtracted)
    l_mes        float      measured suspension-to-load distance
"""

import cv2
import numpy as np
import pickle
import threading
import time

# ---------------- configuration ----------------
CAMERA_ID  = 0
CALIB_FILE = "output/calibration_data.pkl"

ID_CHARGE = 8                   # crate, hanging
ID_HELPER = 12                  # helper on the floor, measured
ID_REF    = 5                   # new marker, static anchor

TAILLE_CHARGE = 0.157           # printed side [m]
TAILLE_HELPER = 0.100
TAILLE_REF    = 0.157

# Offsets, unchanged from the previous experiments. OFFSET_ACCROCHE is on the
# crate; the four below are on the helper, and are applied to its frozen pose.
OFFSET_ACCROCHE = np.array([-0.05, 0.12, 0.0])    # marker 8 -> cable attachment
OFFSET_ORIGINE  = np.array([-0.07, 0.095, 0.0])   # helper   -> world origin
CRANE_TARGET    = np.array([-0.15, -0.17, 0.0])   # origin   -> point aimed at
CIBLE_DEMI      = np.array([0.030, 0.030])
OFFSET_LONG_SIDE_ALIGNED  = np.array([-0.155, -0.225, 0.0])
OFFSET_SHORT_SIDE_ALIGNED = np.array([-0.250, -0.160, 0.0])

# Camera horizontal axes -> robot base axes. From calib_cam2base.py.
CAM2BASE = np.array([[1.0, 0.0],
                     [0.0, -1.0]])

RETARD_CAM  = 0.031             # capture -> availability [s]
N_CALIB     = 60                # frames for the suspension and the frozen pose
ALPHA_REF   = 0.5               # low-pass on the ID_REF frame
SEUIL_BOUGE = 0.005             # displacement above which the helper has moved [m]

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


def _moyenne_rotation(liste):
    """Average rotations, then project back onto SO(3)."""
    U, _, Vt = np.linalg.svd(np.mean(liste, axis=0))
    return U @ Vt


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
    p_5f, R_5f = None, None                     # smoothed ID_REF frame
    t_h0, R_h0, gel = None, None, []            # frozen helper pose, in ID_REF

    while not _stop.is_set():
        ok, frame = cap.read()
        t = time.perf_counter() - RETARD_CAM
        if not ok:
            continue

        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        charge = _pose(corners, ids, ID_CHARGE, TAILLE_CHARGE, mtx, dist)
        helper = _pose(corners, ids, ID_HELPER, TAILLE_HELPER, mtx, dist)
        ref = _pose(corners, ids, ID_REF, TAILLE_REF, mtx, dist)
        etat["vus"] = (ref is not None, charge is not None, helper is not None)

        # ---- the static frame, low-pass filtered ----
        if ref is not None:
            p_r, R_r = ref
            if p_5f is None:
                p_5f, R_5f = p_r, R_r
            else:
                p_5f = (1 - ALPHA_REF) * p_5f + ALPHA_REF * p_r
                R_5f = (1 - ALPHA_REF) * R_5f + ALPHA_REF * R_r
                U, _, Vt = np.linalg.svd(R_5f)
                R_5f = U @ Vt

        # ---- helper, expressed in the ID_REF frame ----
        pos_h = None
        if helper is not None and p_5f is not None:
            p_h, R_h = helper
            pos_h = R_5f.T @ (p_h - p_5f)       # helper origin, ID_REF axes
            R_h_ref = R_5f.T @ R_h              # helper orientation, ID_REF axes

            if t_h0 is None:
                gel.append((pos_h, R_h_ref))
                if len(gel) >= N_CALIB:
                    t_h0 = np.mean([g[0] for g in gel], axis=0)
                    R_h0 = _moyenne_rotation([g[1] for g in gel])
                    print(f"\npose du helper gelee: {np.round(1000*t_h0, 1)} mm "
                          f"dans le repere du marqueur {ID_REF}")
            else:
                d_ref = pos_h - t_h0                    # in ID_REF axes
                depl = (R_h0.T @ d_ref)[:2]             # in the anchor axes
                dR = R_h0.T @ R_h_ref
                dyaw = float(np.arctan2(dR[1, 0], dR[0, 0]))
                etat["helper_pos"] = pos_h[:2]
                etat["helper_depl"] = depl
                etat["helper_dyaw"] = dyaw
                etat["helper_bouge"] = bool(np.linalg.norm(depl) > SEUIL_BOUGE)
            etat["helper_pret"] = t_h0 is not None

        if charge is None:
            continue
        p_c, R_c = charge
        accroche = p_c + R_c @ OFFSET_ACCROCHE

        # ---- suspension calibration: load at rest, cable vertical ----
        if suspension is None:
            calib.append(accroche)
            if len(calib) >= N_CALIB:
                suspension = np.mean(calib, axis=0) - np.array([0.0, 0.0, l_cable])
                print(f"\nsuspension calibree: {np.round(suspension, 4)}")
            continue

        # ---- raw sway angles ----
        c = accroche - suspension
        th = np.array([np.arctan2(c[0], c[2]), np.arctan2(c[1], c[2])])
        etat["theta"] = CAM2BASE @ th
        etat["l_mes"] = float(np.linalg.norm(c))

        # ---- geometry, anchored to the frozen helper pose ----
        # The anchor frame is where the helper was at startup. Bringing it back
        # into camera coordinates costs one composition, and buys a target that
        # stays put when the helper is hit.
        coin = cible = zones = None
        dans, zone_ok = False, None
        if t_h0 is not None and p_5f is not None:
            R_anc = R_5f @ R_h0                             # camera <- anchor
            o_anc = p_5f + R_5f @ t_h0                      # anchor origin, camera

            coin = o_anc + R_anc @ OFFSET_ORIGINE
            cible = coin + R_anc @ CRANE_TARGET
            etat["erreur"] = CAM2BASE @ (cible - accroche)[:2]

            zones = {"long":  coin + R_anc @ OFFSET_LONG_SIDE_ALIGNED,
                     "short": coin + R_anc @ OFFSET_SHORT_SIDE_ALIGNED}
            err_monde = None
            for nom, centre in zones.items():
                e = (R_anc.T @ (accroche - centre))[:2]
                if err_monde is None or np.linalg.norm(e) < np.linalg.norm(err_monde):
                    err_monde = e
                if np.all(np.abs(e) < CIBLE_DEMI):
                    dans, zone_ok = True, nom
            etat["err_monde"] = err_monde
            etat["dans_cible"] = dans
            etat["zone"] = zone_ok

            x_c = R_anc.T @ R_c[:, 0]
            etat["yaw"] = float(np.arctan2(x_c[1], x_c[0]))
            etat["charge_monde"] = (R_anc.T @ (accroche - coin))[:2]

        etat["t"] = t
        etat["pret"] = "erreur" in etat and etat.get("helper_pret", False)

        # ---- debug window ----
        if AFFICHAGE:
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            u_a = _triede(frame, accroche, R_c, mtx, dist)
            cv2.circle(frame, u_a, 6, (0, 165, 255), 2)
            cv2.putText(frame, "accroche", (u_a[0] + 14, u_a[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

            if zones is not None:
                u_c = _triede(frame, coin, R_anc, mtx, dist)
                cv2.circle(frame, u_c, 6, (255, 255, 0), 2)
                cv2.putText(frame, "origine (gelee)", (u_c[0] + 14, u_c[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                            cv2.LINE_AA)
                w, h = CIBLE_DEMI
                for nom, centre in zones.items():
                    sommets = [centre + R_anc @ np.array([sx * w, sy * h, 0.0])
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

            if ref is not None:
                u_5 = _triede(frame, p_5f, R_5f, mtx, dist, longueur=0.05)
                cv2.putText(frame, f"ref ({ID_REF})", (u_5[0] + 12, u_5[1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                            cv2.LINE_AA)

            # helper: frozen pose as a grey cross, current pose, and the gap
            if helper is not None and t_h0 is not None:
                p_h_cam = p_5f + R_5f @ pos_h
                p_h0_cam = p_5f + R_5f @ t_h0
                u_h = _projette(p_h_cam, mtx, dist)
                u_r = _projette(p_h0_cam, mtx, dist)
                bouge = etat.get("helper_bouge", False)
                couleur = (0, 0, 255) if bouge else (255, 0, 255)
                cv2.drawMarker(frame, u_r, (150, 150, 150),
                               cv2.MARKER_TILTED_CROSS, 16, 1)
                cv2.circle(frame, u_h, 7, couleur, 2)
                cv2.line(frame, u_r, u_h, couleur, 2, cv2.LINE_AA)
                d = etat.get("helper_depl", np.zeros(2))
                cv2.putText(frame, f"helper {1000*np.linalg.norm(d):.0f} mm",
                            (u_h[0] + 14, u_h[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2, cv2.LINE_AA)

            e = etat.get("erreur", np.zeros(2))
            em = etat.get("err_monde", np.zeros(2))
            hd = etat.get("helper_depl", np.zeros(2))
            lignes = [
                f"error base   {1000*e[0]:+6.0f}, {1000*e[1]:+6.0f} mm",
                f"error world  {1000*em[0]:+6.0f}, {1000*em[1]:+6.0f} mm  "
                f"{'IN (' + str(zone_ok) + ')' if dans else 'out'}",
                f"theta        {np.degrees(etat['theta'][0]):+5.1f}, "
                f"{np.degrees(etat['theta'][1]):+5.1f} deg",
                f"HELPER depl  {1000*hd[0]:+6.1f}, {1000*hd[1]:+6.1f} mm  "
                f"|d| {1000*np.linalg.norm(hd):5.1f} mm  "
                f"dyaw {np.degrees(etat.get('helper_dyaw', 0.0)):+5.1f} deg"
                f"{'   A BOUGE' if etat.get('helper_bouge') else ''}",
                f"l_mes        {etat['l_mes']:.3f} m  (expected {l_cable:.3f})",
                f"seen         ref({ID_REF}) {ref is not None}   "
                f"charge({ID_CHARGE}) True   "
                f"helper({ID_HELPER}) {helper is not None}",
            ]
            for i, s in enumerate(lignes):
                y = 28 + 26 * i
                cv2.putText(frame, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255) if i == 3 and etat.get("helper_bouge")
                            else (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("vision 8", frame)
            cv2.waitKey(1)

    cap.release()
    if AFFICHAGE:
        cv2.destroyAllWindows()


def start(etat, l_cable):
    etat.update({"theta": np.zeros(2), "err_monde": np.zeros(2), "yaw": 0.0,
                 "charge_monde": np.zeros(2), "zone": None,
                 "helper_pos": np.zeros(2), "helper_depl": np.zeros(2),
                 "helper_dyaw": 0.0, "helper_bouge": False,
                 "helper_pret": False, "vus": (False, False, False),
                 "dans_cible": False, "pret": False,
                 "t": time.perf_counter(), "l_mes": l_cable})
    threading.Thread(target=_loop, args=(etat, l_cable), daemon=True).start()


def stop():
    _stop.set()