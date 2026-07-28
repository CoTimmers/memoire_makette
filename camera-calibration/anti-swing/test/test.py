# -*- coding: utf-8 -*-
"""
Test de connexion au robot UR10 (ou a URSim) - LECTURE SEULE, aucun mouvement.

Usage :
    python connect_test.py                  -> se connecte a URSim (127.0.0.1)
    python connect_test.py 192.168.0.100    -> se connecte au vrai robot

Prerequis : le robot (ou URSim) doit etre demarre : bouton ON puis START.
"""

import sys
import time
from rtde_receive import RTDEReceiveInterface

IP = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"

print(f"Connexion a {IP} ...")
r = RTDEReceiveInterface(IP)
print("Connecte.\n")

print("pose TCP [x, y, z, rx, ry, rz] :", [round(v, 4) for v in r.getActualTCPPose()])
print("vitesse TCP                    :", [round(v, 4) for v in r.getActualTCPSpeed()])
print("angles articulaires            :", [round(v, 4) for v in r.getActualQ()])

# Suivi pendant 10 s : bouge le robot a la main (freedrive) ou avec le pendant,
# et regarde quelle composante change -> c'est ton axe de translation (AXE).
print("\nSuivi 10 s - bouge le robot et observe quelle colonne varie :")
print("     t       x        y        z")
t0 = time.time()
while time.time() - t0 < 10:
    p = r.getActualTCPPose()
    print(f"  {time.time()-t0:5.1f}  {p[0]:+7.4f}  {p[1]:+7.4f}  {p[2]:+7.4f}")
    time.sleep(0.5)

print("\nTermine. Aucun mouvement n'a ete commande.")


