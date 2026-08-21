import time
import numpy as np
import vision

L_CABLE = 1.08

vision.AFFICHAGE = True

etat = {}

vision.start(etat, L_CABLE)

print("Waiting for vision calibration...")

try:

    while not etat.get("pret", False):
        time.sleep(0.1)

    print("Ready. Ctrl-C to stop.\n")

    while True:

        x = etat.get("attach_x", 0.0)
        y = etat.get("attach_y", 0.0)
        yaw = np.degrees(etat.get("yaw_ref", 0.0))

        funnels = etat.get("funnels", {})

        active = [
            name
            for name, inside in funnels.items()
            if inside
        ]

        active_text = ", ".join(active) if active else "NONE"

        print(
            f"\r"
            f"attachment = ({1000*x:+6.0f}, {1000*y:+6.0f}) mm | "
            f"yaw = {yaw:+6.1f} deg | "
            f"funnel = {active_text}",
            end="",
            flush=True
        )

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

finally:
    vision.stop()
    print("\nstopped.")