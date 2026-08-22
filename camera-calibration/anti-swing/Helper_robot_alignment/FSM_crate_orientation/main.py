import time
import csv
import threading
import numpy as np
import rtde_control
import rtde_receive
import vision


ROBOT_IP = "192.168.56.102"

V = 0.08
A = 0.2

L_CABLE = 1.08

PERIODE_LOG = 0.01


# ==================================================
# Geometrie
# ==================================================

# Centre de l'ArUco helper, dans le repere base robot [m]
T_MARKER_IN_BASE = np.array([0.325, -0.067, 0.124])

# Base robot -> axes du marqueur helper.
# Identite tant que le test de deplacement n'a pas montre le contraire.
R_BASE_TO_MARKER = np.array([
    [-0.274250, -0.961658, 0.0],
    [+0.961658, -0.274250, 0.0],
    [ 0.0,       0.0,       1.0],
])



def tcp_dans_helper(tcp_pose):
    """TCP (pose RTDE, metres) exprime dans le repere ArUco helper."""
    p_base = np.array(tcp_pose[:3])
    return R_BASE_TO_MARKER @ (p_base - T_MARKER_IN_BASE)


# ==================================================
# Robot
# ==================================================

rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)

pose_start = rtde_r.getActualTCPPose()

# Hauteur et orientation figees sur la pose actuelle
Z_FIXE = pose_start[2]
RX, RY, RZ = pose_start[3], pose_start[4], pose_start[5]

print(f"Z fige a {1000*Z_FIXE:.1f} mm")


# ==================================================
# Poses [mm]
# ==================================================

POSES_MM = [
    ("home",             530,   39),
    ("engage_pivoting",  530, -141), #Avant de changer les region acceptable
    # ("engage_pivoting",  530, -175),
    ("finish_pivoting",  530, -293),
    # ("finish_pivoting",  530, -300),
    ("shift_1",          595, -293),
    ("shift_2",          595,   23),
    ("shift_3",          565,  -33),
]


def pose_xy(x_mm, y_mm):
    """Pose RTDE complete a partir d'un XY en mm."""
    return [
        x_mm / 1000.0,
        y_mm / 1000.0,
        Z_FIXE,
        RX,
        RY,
        RZ,
    ]


SEQUENCE = [
    (nom, pose_xy(x, y))
    for nom, x, y in POSES_MM
]


# ==================================================
# Vision
# ==================================================

etat = {}

vision.AFFICHAGE = True
vision.start(etat, L_CABLE)

print("Waiting for vision...")

while not etat.get("pret", False):
    time.sleep(0.1)

print("Vision ready.")


# ==================================================
# CSV + thread de log
# ==================================================

log = open("fsm_test.csv", "w", newline="")
writer = csv.writer(log)

writer.writerow([
    "t",
    "etape",
    "en_mouvement",

    "attach_x",
    "attach_y",
    "yaw_ref",

    "tcp_base_x",
    "tcp_base_y",

    "tcp_helper_x",
    "tcp_helper_y",
])


etape_courante = "init"
en_mouvement = 0
stop_log = threading.Event()
verrou = threading.Lock()

t0 = time.perf_counter()


def boucle_log():

    while not stop_log.is_set():

        t = time.perf_counter() - t0

        tcp = rtde_r.getActualTCPPose()
        tcp_h = tcp_dans_helper(tcp)

        with verrou:
            writer.writerow([
                f"{t:.4f}",

                etape_courante,
                en_mouvement,

                f"{etat.get('attach_x', 0.0):.5f}",
                f"{etat.get('attach_y', 0.0):.5f}",
                f"{etat.get('yaw_ref', 0.0):.5f}",

                f"{tcp[0]:.5f}",
                f"{tcp[1]:.5f}",

                f"{tcp_h[0]:.5f}",
                f"{tcp_h[1]:.5f}",
            ])

        time.sleep(PERIODE_LOG)


threading.Thread(
    target=boucle_log,
    daemon=True
).start()


# ==================================================
# Sequence pilotee au clavier
# ==================================================

try:

    for i, (nom, cible) in enumerate(SEQUENCE):

        print(
            f"\n[{i+1}/{len(SEQUENCE)}] {nom}"
            f"   x={1000*cible[0]:+6.1f}  y={1000*cible[1]:+6.1f} mm"
        )

        input("    Enter pour lancer (Ctrl-C pour arreter) ...")

        etape_courante = nom
        en_mouvement = 1

        rtde_c.moveL(cible, V, A)

        en_mouvement = 0

        tcp = rtde_r.getActualTCPPose()
        tcp_h = tcp_dans_helper(tcp)

        print(
            f"    atteint   "
            f"tcp_base = ({1000*tcp[0]:+6.1f}, {1000*tcp[1]:+6.1f}) mm | "
            f"tcp_helper = ({1000*tcp_h[0]:+6.1f}, {1000*tcp_h[1]:+6.1f}) mm"
        )

        print(
            f"    attach    "
            f"({1000*etat.get('attach_x', 0.0):+6.1f}, "
            f"{1000*etat.get('attach_y', 0.0):+6.1f}) mm | "
            f"yaw = {np.degrees(etat.get('yaw_ref', 0.0)):+6.1f} deg"
        )

    print("\nSequence terminee.")


except KeyboardInterrupt:

    print("\nStopped.")


finally:

    stop_log.set()
    time.sleep(0.2)

    rtde_c.stopL()

    with verrou:
        log.close()

    vision.stop()

    rtde_c.stopScript()

    print("CSV saved.")