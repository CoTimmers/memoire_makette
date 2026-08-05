import numpy as np, time
import vision

L_CABLE = 1.25

etat = {}
vision.start(etat, L_CABLE)
print("bac immobile pour la calibration du pivot...")

try:
    while not etat["pret"]:
        time.sleep(0.1)
    print("pret.\n")
    while True:
        th = np.degrees(etat["theta"])
        print(f"\rtheta {th[0]:+6.2f}, {th[1]:+6.2f} deg | "
              f"l {etat['l_mes']:.3f} m | vu {int(etat['vu'])}",
              end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    vision.stop()
    print("\nstop.")