"""Check 7_vision.py alone. No robot, nothing moves.

Run this before every bench session. It starts the real vision thread, so what
you see here is exactly what the controller will see during a run.

Offsets are edited by hand in 7_vision.py, then this is relaunched.

What to verify, in order

  1. vus 11: both markers are seen from the working pose. If marker 8 leaves the
     field of view during a run, 7_main aborts after 2 s, so check the framing
     here first, at both D1 and D3.

  2. the orange dot sits on the cable attachment, and the cyan frame sits on the
     world origin. If not, correct OFFSET_PIVOT or OFFSET_ORIGINE and relaunch.

  3. l_mes stays at the cable length. A constant offset means a wrong printed
     size or a wrong OFFSET_PIVOT; a slow drift means the suspension was
     calibrated while the load was still moving.

  4. theta is near zero at rest and has the sign you expect: push the load
     towards base +X and theta[0] must go positive. This is the CAM2BASE check.
     Getting it wrong makes the feedback amplify the sway instead of damping it,
     and you would discover it during the return phase, with 2.5 deg already
     swinging.

  5. D1 and D3 are drawn where you want them, on the 45 degree diagonal, D3 just
     short of the crate. Carry the load over each by hand and read err D1 and
     err D3: they must go to zero there.

  6. age stays under about 60 ms, and dist origine agrees with a tape measure.
"""

import numpy as np
import time
from importlib import import_module

vision = import_module("7_vision")

L_CABLE = 1.11                  # same value as 7_main.py [m]

vision.AFFICHAGE = True         # debug window
vision.CIBLE_ACTIVE = "d3"      # which point is drawn in green

etat = {}
vision.start(etat, L_CABLE)
print(f"OFFSET_PIVOT   {vision.OFFSET_PIVOT.tolist()}")
print(f"OFFSET_ORIGINE {vision.OFFSET_ORIGINE.tolist()}")
print(f"cable {L_CABLE} m. Load hanging still, cable vertical, for the "
      f"suspension calibration ({vision.N_CALIB} frames).")

try:
    while not etat["pret"]:
        time.sleep(0.1)
    print("ready. Ctrl-C to stop.\n")
    while True:
        e1 = etat["err_d1"]
        e3 = etat["err_d3"]
        pos = etat["pos_monde"]
        th = np.degrees(etat["theta"])
        ref, charge = etat["vus"]
        age = 1000 * (time.perf_counter() - etat["t"])
        print(f"\rD1 {1000*np.linalg.norm(e1):5.0f} mm | "
              f"D3 {1000*np.linalg.norm(e3):5.0f} mm | "
              f"monde {1000*pos[0]:+6.0f},{1000*pos[1]:+6.0f} mm "
              f"(bac {1000*etat['dist_origine']:5.0f}) | "
              f"th {th[0]:+5.1f},{th[1]:+5.1f} deg | "
              f"yaw {np.degrees(etat['yaw']):+6.1f} | "
              f"l {etat['l_mes']:.3f} m | "
              f"vus {int(ref)}{int(charge)} | age {age:4.0f} ms",
              end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    vision.stop()
    print("\nstopped.")