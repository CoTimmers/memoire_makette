"""Check vision.py alone. No robot, nothing moves.

Run this before every session on the bench. What to verify, in order:

  1. vus = (1, 1): both markers are seen from the start pose. If the reference
     leaves the field of view during the run the main loop aborts, so check the
     framing here first.

  2. l_mes stays at the cable length. A constant offset means a wrong marker
     size or a wrong OFFSET_ACCROCHE; a drift means the pivot calibration was
     done while the load was still moving.

  3. theta is near zero at rest, and changes sign the way you expect: push the
     crate along the base X axis and theta[0] must follow that sign. This is
     the CAM2BASE check, and getting it wrong makes the feedback divergent.

  4. the rectangle in the window is where you want the load to end up, and it
     turns green when you carry the crate into it by hand.

  5. age stays under about 60 ms. Above that the camera is dropping frames and
     the Kalman correction lands too far in the past.
"""

import numpy as np
import time
import vision

L_CABLE = 1.08                   # cable length [m], same value as main_vision.py

vision.AFFICHAGE = True          # debug window

etat = {}
vision.start(etat, L_CABLE)
print(f"cable {L_CABLE} m. Load still, cable vertical, for the pivot "
      f"calibration ({vision.N_CALIB} frames).")

try:
    while not etat["pret"]:
        time.sleep(0.1)
    print("ready. Ctrl-C to stop.\n")
    while True:
        e = etat["erreur"]
        em = etat["err_monde"]
        th = np.degrees(etat["theta"])
        ref, charge = etat["vus"]
        age = 1000 * (time.perf_counter() - etat["t"])
        print(f"\re {1000*e[0]:+6.0f},{1000*e[1]:+6.0f} mm | "
              f"monde {1000*em[0]:+6.0f},{1000*em[1]:+6.0f} mm "
              f"{'IN ' if etat['dans_cible'] else 'out'} | "
              f"th {th[0]:+5.1f},{th[1]:+5.1f} deg | "
              f"l {etat['l_mes']:.3f} m | "
              f"vus {int(ref)}{int(charge)} | age {age:4.0f} ms",
              end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    vision.stop()
    print("\nstopped.")