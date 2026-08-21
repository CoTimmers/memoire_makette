import time
import csv
import rtde_control
import rtde_receive
import vision


ROBOT_IP = "192.168.56.102"

V = 0.08
A = 0.2

L_CABLE = 1.08


# ==================================================
# Robot
# ==================================================

rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)

pose_start = rtde_r.getActualTCPPose()


# ==================================================
# Hardcoded actions
# ==================================================

POSE_1 = list(pose_start)
POSE_1[0] += 0.10

POSE_2 = list(pose_start)
POSE_2[1] -= 0.10

POSE_3 = list(pose_start)
POSE_3[0] -= 0.05


ACTIONS = {
    "MODE_1": POSE_1,
    "MODE_2": POSE_2,
    "MODE_3": POSE_3,
}


MODE_SEQUENCE = [
    "MODE_1",
    "MODE_2",
    "MODE_3",
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
# CSV
# ==================================================

log = open("fsm_test.csv", "w", newline="")
writer = csv.writer(log)

writer.writerow([
    "t",
    "mode",
    "action_triggered",

    "attach_x",
    "attach_y",
    "yaw_ref",

    "tcp_x",
    "tcp_y",

    "mode_1",
    "mode_2",
    "mode_3",
])


# ==================================================
# FSM
# ==================================================

mode_index = 0

t0 = time.perf_counter()


try:

    while mode_index < len(MODE_SEQUENCE):

        t = time.perf_counter() - t0

        mode = MODE_SEQUENCE[mode_index]

        funnels = etat.get("funnels", {})

        triggered = 0


        # ------------------------------------------
        # Entry state reached
        # ------------------------------------------

        if funnels.get(mode, False):

            print(f"\nENTRY STATE {mode}")

            triggered = 1

            target = ACTIONS[mode]

            print(f"Action {mode}")

            rtde_c.moveL(
                target,
                V,
                A,
                asynchronous=True
            )

            mode_index += 1


        # ------------------------------------------
        # Measurements
        # ------------------------------------------

        tcp = rtde_r.getActualTCPPose()

        writer.writerow([
            f"{t:.4f}",

            mode,
            triggered,

            f"{etat.get('attach_x', 0.0):.5f}",
            f"{etat.get('attach_y', 0.0):.5f}",
            f"{etat.get('yaw_ref', 0.0):.5f}",

            f"{tcp[0]:.5f}",
            f"{tcp[1]:.5f}",

            int(funnels.get("MODE_1", False)),
            int(funnels.get("MODE_2", False)),
            int(funnels.get("MODE_3", False)),
        ])

        time.sleep(0.01)


except KeyboardInterrupt:

    print("\nStopped.")


finally:

    rtde_c.stopL()

    log.close()

    vision.stop()

    rtde_c.stopScript()

    print("CSV saved.")