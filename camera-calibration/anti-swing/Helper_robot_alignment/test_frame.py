"""Validate frames.py alone. No robot, no motion.

What to check, in order:
  1. both markers are seen;
  2. l_mes stays equal to the cable length (a drift means a wrong marker size
     or a wrong offset);
  3. the red cross lands where you expect the target to be on the crate;
  4. moving the helper by hand towards the cross makes the error go to zero;
  5. the error components change sign the way you expect along the base axes.
"""

import numpy as np
import time
import frames_affichage as frames

L_CABLE = 0.5225                 # cable length [m]
D1 = (0.30, 0.10)                # target in the crate frame [m]

frames.AFFICHAGE = True          # debug window

etat = {}
frames.start(etat, L_CABLE, d1=D1)
print(f"target {D1} m in the crate frame. Keep the helper still for calibration.")

try:
    while not etat["pret"]:
        time.sleep(0.1)
    print("ready.\n")
    while True:
        e = etat["erreur"]
        th = np.degrees(etat["theta"])
        bac, hlp = etat["vus"]
        age = 1000 * (time.perf_counter() - etat["t"])
        print(f"\rerror {1000*e[0]:+6.0f}, {1000*e[1]:+6.0f} mm | "
              f"|e| {1000*np.linalg.norm(e):5.0f} mm | "
              f"theta {th[0]:+5.1f}, {th[1]:+5.1f} deg | "
              f"l {etat['l_mes']:.3f} m | crate {int(bac)} helper {int(hlp)} | "
              f"age {age:4.0f} ms", end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    frames.stop()
    print("\nstopped.")