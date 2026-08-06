"""Check 8_vision.py alone. No robot, nothing moves.

Run before every session. It starts the real vision thread, so what you see is
what the controller will see.

What to verify, in order

  1. vus 111: the three markers are seen from the working pose, and stay seen
     over the whole path. Marker 5 is the one that matters most: lose it and the
     helper measurement loses its frame.

  2. the cyan frame sits on the world origin you chose on marker 5. This is a
     new marker, so OFFSET_ORIGINE has to be measured; it is not the old one.

  3. CRANE_TARGET and the two drop zones are where you want them. They were
     measured relative to marker 12 back when it was the reference. Marker 5 is
     somewhere else, so these numbers are almost certainly wrong until you
     correct them here.

  4. the helper displacement reads near zero and stays there while nothing is
     touched. That is the noise floor of the measurement. Whatever it drifts to
     over a minute is the resolution you can claim; SEUIL_BOUGE must sit above
     it.

  5. push the helper by hand a known distance, say 50 mm against a ruler, and
     check the reading. This calibrates the whole chain in one gesture.

  6. l_mes stays at the cable length, and theta has the sign you expect.
"""

import numpy as np
import time
from importlib import import_module

vision = import_module("8_vision")

L_CABLE = 1.11

vision.AFFICHAGE = True

etat = {}
vision.start(etat, L_CABLE)
print(f"OFFSET_ACCROCHE {vision.OFFSET_ACCROCHE.tolist()}   sur le marqueur "
      f"{vision.ID_CHARGE}")
print(f"OFFSET_HELPER   {vision.OFFSET_HELPER.tolist()}   sur le marqueur "
      f"{vision.ID_HELPER}")
print(f"OFFSET_ORIGINE  {vision.OFFSET_ORIGINE.tolist()}   sur le marqueur "
      f"{vision.ID_REF}")
print(f"seuil de mouvement du helper: {1000*vision.SEUIL_BOUGE:.0f} mm")
print("tout immobile pour les calibrations...")

try:
    while not etat["pret"]:
        time.sleep(0.1)
    print("ready. Ctrl-C to stop.\n")
    depl_max = 0.0
    while True:
        e = etat["erreur"]
        em = etat["err_monde"]
        th = np.degrees(etat["theta"])
        d = etat["helper_depl"]
        n_d = float(np.linalg.norm(d))
        depl_max = max(depl_max, n_d)
        ref, charge, helper = etat["vus"]
        age = 1000 * (time.perf_counter() - etat["t"])
        print(f"\re {1000*np.linalg.norm(e):5.0f} mm | "
              f"monde {1000*em[0]:+6.0f},{1000*em[1]:+6.0f} "
              f"{'IN ' + str(etat['zone']) if etat['dans_cible'] else 'out  '} | "
              f"th {th[0]:+5.1f},{th[1]:+5.1f} | "
              f"HELPER {1000*d[0]:+6.1f},{1000*d[1]:+6.1f} mm "
              f"|d| {1000*n_d:5.1f} (max {1000*depl_max:5.1f}) "
              f"dyaw {np.degrees(etat['helper_dyaw']):+5.1f} "
              f"{'BOUGE' if etat['helper_bouge'] else '     '} | "
              f"vus {int(ref)}{int(charge)}{int(helper)} | {age:4.0f} ms",
              end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    vision.stop()
    print(f"\nderive maximale du helper sans rien toucher: "
          f"{1000*depl_max:.1f} mm")
    print(f"SEUIL_BOUGE vaut {1000*vision.SEUIL_BOUGE:.0f} mm, "
          f"il doit rester nettement au-dessus.")
    print("stopped.")