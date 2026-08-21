"""
Vision for FSM experiments.

Two ArUco markers:

    ID_CHARGE = marker on the suspended crate
    ID_REF    = marker on the helper robot, static reference

The vision thread provides:

    erreur          [ex, ey]
        Error from crate attachment point to CRANE_TARGET,
        expressed in robot base axes [m]

    theta           [thx, thy]
        Raw sway angles in robot base axes [rad]

    yaw
        Crate yaw relative to helper/world frame [rad]

    attach_x
    attach_y
        Position of the crate cable attachment point expressed
        directly in the helper ArUco frame [m]

    yaw_ref
        Crate yaw relative to helper ArUco frame [rad]

    funnels
        Dictionary:
        {
            "MODE_1": True/False,
            "MODE_2": True/False,
            ...
        }

    active_funnels
        List of funnels currently satisfied

    dans_cible
        Existing target-zone test

    vus
        (reference_seen, crate_seen)

    pret
        True when calibration is complete and measurements exist

    t
        Vision timestamp

    l_mes
        Measured suspension length
"""

import cv2
import numpy as np
import pickle
import threading
import time


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_ID = 0
CALIB_FILE = "output/calibration_data.pkl"

ID_CHARGE = 8
ID_REF = 12

TAILLE_CHARGE = 0.157
TAILLE_REF = 0.100


# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------

# Reference marker -> helper/world origin
OFFSET_ORIGINE = np.array([
     0.080,
     0.065,
     0.0
])

# Crate marker -> cable attachment point
OFFSET_ACCROCHE = np.array([
    -0.01,
     0.09,
     0.0
])

# Existing target used by previous controller
CRANE_TARGET = np.array([
    -0.15,
    -0.17,
     0.0
])

CIBLE_DEMI = np.array([
    0.030,
    0.030
])

OFFSET_LONG_SIDE_ALIGNED = np.array([
    -0.155,
    -0.225,
     0.0
])

OFFSET_SHORT_SIDE_ALIGNED = np.array([
    -0.250,
    -0.160,
     0.0
])


# ------------------------------------------------------------
# Camera horizontal axes -> robot base axes
# ------------------------------------------------------------

CAM2BASE = np.array([
    [1.0,  0.0],
    [0.0, -1.0]
])


RETARD_CAM = 0.031
N_CALIB = 60

AFFICHAGE = False


# ============================================================
# FUNNELS
# ============================================================

"""
IMPORTANT

x and y below are coordinates of the CRATE ATTACHMENT POINT
expressed directly in the ArUco frame of the helper robot.

yaw is the crate orientation relative to the helper robot.

All positions are in metres.
All angles are internally stored in radians.

Change these values according to your real modes.
"""

FUNNELS = {

    "MODE_1": {

        "x_min": -0.35,
        "x_max": -0.25,

        "y_min": -0.15,
        "y_max": -0.05,

        "yaw_min": np.radians(-15),
        "yaw_max": np.radians(15),
    },

    "MODE_2": {

        "x_min": -0.25,
        "x_max": -0.15,

        "y_min": -0.15,
        "y_max": -0.05,

        "yaw_min": np.radians(-10),
        "yaw_max": np.radians(10),
    },

    "MODE_3": {

        "x_min": -0.15,
        "x_max": -0.05,

        "y_min": -0.10,
        "y_max": 0.00,

        "yaw_min": np.radians(-5),
        "yaw_max": np.radians(5),
    },
}


# ============================================================
# INTERNAL STATE
# ============================================================

_stop = threading.Event()


# ============================================================
# FUNNEL FUNCTIONS
# ============================================================

def angle_wrap(angle):
    """Return angle between -pi and +pi."""
    return np.arctan2(
        np.sin(angle),
        np.cos(angle)
    )


def angle_inside(yaw, yaw_min, yaw_max):
    """
    Test yaw interval.

    This simple version assumes the interval does not cross +/-180 deg.
    """
    yaw = angle_wrap(yaw)

    return yaw_min <= yaw <= yaw_max


def inside_funnel(x, y, yaw, funnel):
    """
    Check whether the crate state belongs to one funnel.
    """

    xy_ok = (
        funnel["x_min"] <= x <= funnel["x_max"]
        and
        funnel["y_min"] <= y <= funnel["y_max"]
    )

    yaw_ok = angle_inside(
        yaw,
        funnel["yaw_min"],
        funnel["yaw_max"]
    )

    return bool(xy_ok and yaw_ok)


# ============================================================
# VISION UTILITIES
# ============================================================

def _projette(p, mtx, dist):

    pt, _ = cv2.projectPoints(
        np.asarray(
            p,
            dtype=np.float64
        ).reshape(1, 3),

        np.zeros(3),
        np.zeros(3),

        mtx,
        dist
    )

    return (
        int(pt[0][0][0]),
        int(pt[0][0][1])
    )


def _obj_points(taille):

    h = taille / 2

    return np.array([
        [-h,  h, 0],
        [ h,  h, 0],
        [ h, -h, 0],
        [-h, -h, 0]
    ],
        dtype=np.float32
    )


def _pose(
    corners,
    ids,
    cible_id,
    taille,
    mtx,
    dist
):

    if ids is None:
        return None

    idx = np.where(
        ids.flatten() == cible_id
    )[0]

    if len(idx) == 0:
        return None

    ok, rvec, tvec = cv2.solvePnP(
        _obj_points(taille),
        corners[idx[0]][0],
        mtx,
        dist
    )

    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)

    return (
        tvec.flatten(),
        R
    )


def _triede(
    frame,
    origine,
    R,
    mtx,
    dist,
    longueur=0.06
):

    o = _projette(
        origine,
        mtx,
        dist
    )

    axes = [
        ((longueur, 0, 0),
         (0, 0, 255),
         "X"),

        ((0, longueur, 0),
         (0, 255, 0),
         "Y"),

        ((0, 0, longueur),
         (255, 0, 0),
         "Z")
    ]

    for vec, couleur, nom in axes:

        p = _projette(
            origine + R @ np.array(vec),
            mtx,
            dist
        )

        cv2.line(
            frame,
            o,
            p,
            couleur,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            nom,
            p,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            couleur,
            1,
            cv2.LINE_AA
        )

    return o


# ============================================================
# MAIN VISION LOOP
# ============================================================

def _loop(etat, l_cable):

    # --------------------------------------------------------
    # Camera calibration
    # --------------------------------------------------------

    with open(CALIB_FILE, "rb") as f:
        data = pickle.load(f)

    mtx = np.array(
        data.get(
            "camera_matrix",
            data.get("mtx")
        )
    )

    dist = np.array(
        data.get(
            "distortion_coefficients",
            data.get("dist")
        )
    )


    # --------------------------------------------------------
    # ArUco detector
    # --------------------------------------------------------

    params = cv2.aruco.DetectorParameters()

    params.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_SUBPIX
    )

    detector = cv2.aruco.ArucoDetector(

        cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        ),

        params
    )


    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_ID,
        cv2.CAP_V4L2
    )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1920
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        1080
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    ok0, f0 = cap.read()
    print("frame:", f0.shape if ok0 else "echec")

    # --------------------------------------------------------
    # Calibration state
    # --------------------------------------------------------

    pivot = None
    calib = []


    # ========================================================
    # LOOP
    # ========================================================

    while not _stop.is_set():

        ok, frame = cap.read()

        t = (
            time.perf_counter()
            - RETARD_CAM
        )

        if not ok:
            continue


        # ----------------------------------------------------
        # Detect markers
        # ----------------------------------------------------

        corners, ids, _ = detector.detectMarkers(

            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )
        )

        charge = _pose(
            corners,
            ids,
            ID_CHARGE,
            TAILLE_CHARGE,
            mtx,
            dist
        )

        ref = _pose(
            corners,
            ids,
            ID_REF,
            TAILLE_REF,
            mtx,
            dist
        )


        etat["vus"] = (
            ref is not None,
            charge is not None
        )


        # Need the crate
        if charge is None:
            continue


        # ----------------------------------------------------
        # Crate pose
        # ----------------------------------------------------

        p_c, R_c = charge

        # Actual cable attachment point
        accroche = (
            p_c
            + R_c @ OFFSET_ACCROCHE
        )


        # ----------------------------------------------------
        # Pivot calibration
        # ----------------------------------------------------

        if pivot is None:

            calib.append(accroche)

            if len(calib) >= N_CALIB:

                pivot = (
                    np.mean(
                        calib,
                        axis=0
                    )
                    - np.array([
                        0.0,
                        0.0,
                        l_cable
                    ])
                )

                print(
                    "\npivot calibre:",
                    np.round(
                        pivot,
                        4
                    )
                )

            continue


        # ----------------------------------------------------
        # Sway angles
        # ----------------------------------------------------

        c = accroche - pivot

        th_camera = np.array([

            np.arctan2(
                c[0],
                c[2]
            ),

            np.arctan2(
                c[1],
                c[2]
            )
        ])


        etat["theta"] = (
            CAM2BASE
            @ th_camera
        )

        etat["l_mes"] = float(
            np.linalg.norm(c)
        )


        # ----------------------------------------------------
        # Helper-reference-frame measurements
        # ----------------------------------------------------

        if ref is not None:

            p_r, R_r = ref

            # origine_ref = (
            #     p_r
            #     + R_r @ OFFSET_ORIGINE
            # )


            # ------------------------------------------------
            # Attachment point in helper ArUco frame
            # ------------------------------------------------

            p_attach_ref = (
                R_r.T
                @ (accroche - p_r)
            )

            x_attach = float(
                p_attach_ref[0]
            )

            y_attach = float(
                p_attach_ref[1]
            )


            etat["attach_x"] = x_attach
            etat["attach_y"] = y_attach


            # ------------------------------------------------
            # Crate orientation relative to helper
            # ------------------------------------------------

            R_crate_ref = (
                R_r.T
                @ R_c
            )

            yaw_ref = float(

                np.arctan2(
                    R_crate_ref[1, 0],
                    R_crate_ref[0, 0]
                )
            )

            yaw_ref = angle_wrap(
                yaw_ref
            )


            etat["yaw_ref"] = yaw_ref

            # Keep old variable name
            etat["yaw"] = yaw_ref


            # ------------------------------------------------
            # Evaluate funnels
            # ------------------------------------------------

            funnel_states = {}

            for name, funnel in FUNNELS.items():

                funnel_states[name] = inside_funnel(

                    x_attach,
                    y_attach,
                    yaw_ref,
                    funnel
                )


            etat["funnels"] = (
                funnel_states
            )


            etat["active_funnels"] = [

                name

                for name, active
                in funnel_states.items()

                if active
            ]


            # ------------------------------------------------
            # Existing helper/world geometry
            # ------------------------------------------------

            coin = (
                p_r
                + R_r @ OFFSET_ORIGINE
            )


            cible = (
                coin
                + R_r @ CRANE_TARGET
            )


            # Existing controller error
            etat["erreur"] = (

                CAM2BASE
                @ (
                    cible
                    - accroche
                )[:2]
            )


            # ------------------------------------------------
            # Existing drop zones
            # ------------------------------------------------

            zones = {

                "long":

                    coin
                    + R_r
                    @ OFFSET_LONG_SIDE_ALIGNED,

                "short":

                    coin
                    + R_r
                    @ OFFSET_SHORT_SIDE_ALIGNED
            }


            err_monde = None
            dans = False
            zone_ok = None


            for nom, centre in zones.items():

                e = (

                    R_r.T
                    @ (
                        accroche
                        - centre
                    )

                )[:2]


                if (
                    err_monde is None
                    or
                    np.linalg.norm(e)
                    <
                    np.linalg.norm(
                        err_monde
                    )
                ):

                    err_monde = e


                if np.all(
                    np.abs(e)
                    <
                    CIBLE_DEMI
                ):

                    dans = True
                    zone_ok = nom


            etat["err_monde"] = (
                err_monde
            )

            etat["dans_cible"] = (
                dans
            )

            etat["zone"] = (
                zone_ok
            )


            # ------------------------------------------------
            # Debug display
            # ------------------------------------------------

            if AFFICHAGE:

                if ids is not None:

                    cv2.aruco.drawDetectedMarkers(
                        frame,
                        corners,
                        ids
                    )


                # --------------------------------------------
                # Draw crate attachment point
                # --------------------------------------------

                u_attach = _projette(
                    accroche,
                    mtx,
                    dist
                )

                cv2.circle(
                    frame,
                    u_attach,
                    7,
                    (0, 165, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "attachment",
                    (
                        u_attach[0] + 12,
                        u_attach[1] - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA
                )




                # --------------------------------------------
                # Draw helper reference frame
                # --------------------------------------------

                u_ref = _triede(
                    frame,
                    p_r,
                    R_r,
                    mtx,
                    dist,
                    longueur=0.06
                )

                cv2.putText(
                    frame,
                    "HELPER REF",
                    (
                        u_ref[0] + 12,
                        u_ref[1] + 18
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA
                )

                # --------------------------------------------
                # Draw helper origin / pivot
                # --------------------------------------------

                u_origine = _projette(
                    coin,
                    mtx,
                    dist
                )

                cv2.circle(
                    frame,
                    u_origine,
                    7,
                    (255, 0, 255),      # magenta, BGR
                    2
                )

                cv2.putText(
                    frame,
                    "origin",
                    (
                        u_origine[0] + 12,
                        u_origine[1] - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA
                )


                vue = cv2.rotate(
                    frame,
                    cv2.ROTATE_90_COUNTERCLOCKWISE
                )


                # --------------------------------------------
                # Text information
                # --------------------------------------------

                active_text = (

                    ", ".join(
                        etat["active_funnels"]
                    )

                    if etat["active_funnels"]

                    else "NONE"
                )


                lines = [

                    (
                        "attach helper frame "
                        f"x={1000*x_attach:+6.0f} "
                        f"y={1000*y_attach:+6.0f} mm"
                    ),

                    (
                        "yaw helper frame "
                        f"{np.degrees(yaw_ref):+6.1f} deg"
                    ),

                    (
                        f"FUNNEL: {active_text}"
                    ),

                    (
                        "theta "
                        f"{np.degrees(etat['theta'][0]):+5.1f}, "
                        f"{np.degrees(etat['theta'][1]):+5.1f} deg"
                    ),

                    (
                        f"l_mes {etat['l_mes']:.3f} m "
                        f"(expected {l_cable:.3f})"
                    ),

                    (
                        f"seen ref={ref is not None} "
                        f"crate=True"
                    )
                ]


                for i, text in enumerate(lines):

                    y = 28 + 27 * i


                    # black background
                    cv2.putText(
                        vue,
                        text,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA
                    )


                    # active funnel highlighted
                    color = (
                        (0, 255, 0)
                        if (
                            i == 2
                            and
                            etat["active_funnels"]
                        )
                        else
                        (0, 200, 255)
                    )


                    cv2.putText(
                        vue,
                        text,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        1,
                        cv2.LINE_AA
                    )


                cv2.imshow(
                    "FSM vision",
                    vue
                )

                cv2.waitKey(1)


        # ----------------------------------------------------
        # Update timestamp / ready state
        # ----------------------------------------------------

        etat["t"] = t

        etat["pret"] = (
            pivot is not None
            and
            ref is not None
            and
            "attach_x" in etat
            and
            "yaw_ref" in etat
        )


    # ========================================================
    # END LOOP
    # ========================================================

    cap.release()

    if AFFICHAGE:
        cv2.destroyAllWindows()


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def start(etat, l_cable):

    global _stop

    # Important if vision is restarted in same Python process
    _stop.clear()


    etat.update({

        "theta":
            np.zeros(2),

        "err_monde":
            np.zeros(2),

        "yaw":
            0.0,

        "yaw_ref":
            0.0,

        "attach_x":
            0.0,

        "attach_y":
            0.0,

        "funnels":
            {
                name: False
                for name in FUNNELS
            },

        "active_funnels":
            [],

        "vus":
            (False, False),

        "dans_cible":
            False,

        "zone":
            None,

        "pret":
            False,

        "t":
            time.perf_counter(),

        "l_mes":
            l_cable
    })


    threading.Thread(
        target=_loop,
        args=(
            etat,
            l_cable
        ),
        daemon=True
    ).start()


def stop():

    _stop.set()