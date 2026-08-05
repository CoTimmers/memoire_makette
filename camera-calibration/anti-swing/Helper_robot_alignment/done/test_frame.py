import numpy as np
import time
import frames_affichage as frames

L_CABLE = 0.5225
D1 = (0.30, 0.10)

frames.AFFICHAGE = True

etat = {}
frames.start(etat, L_CABLE, d1=D1)
print(f"target {D1} m. Keep the load still for calibration.")

try:
    while not etat["pret"]:
        time.sleep(0.1)
    print("ready.\n")
    while True:
        e = etat["erreur"]
        th = np.degrees(etat["theta"])
        ref, charge = etat["vus"]
        age = 1000 * (time.perf_counter() - etat["t"])
        print(f"\rerror {1000*e[0]:+6.0f}, {1000*e[1]:+6.0f} mm | "
              f"|e| {1000*np.linalg.norm(e):5.0f} mm | "
              f"theta {th[0]:+5.1f}, {th[1]:+5.1f} deg | "
              f"l {etat['l_mes']:.3f} m | ref {int(ref)} charge {int(charge)} | "
              f"age {age:4.0f} ms", end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    frames.stop()
    print("\nstopped.")