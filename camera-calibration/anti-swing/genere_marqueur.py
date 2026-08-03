"""Generate an ArUco marker as a PNG ready to print at a chosen physical size.
 
    python genere_marqueur.py 12 100      -> marker id 12, 100 mm side
"""
 
import cv2
import numpy as np
import sys
 
ID = int(sys.argv[1]) if len(sys.argv) > 1 else 12
COTE_MM = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
DPI = 300
MARGE_MM = 10.0                     # white border, needed for detection
 
dico = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
px = int(round(COTE_MM / 25.4 * DPI))
marge = int(round(MARGE_MM / 25.4 * DPI))
 
img = cv2.aruco.generateImageMarker(dico, ID, px)
page = np.full((px + 2 * marge, px + 2 * marge), 255, dtype=np.uint8)
page[marge:marge + px, marge:marge + px] = img
 
nom = f"aruco_{ID}_{int(COTE_MM)}mm.png"
cv2.imwrite(nom, page)
print(f"{nom}  ({px}x{px} px at {DPI} dpi -> {COTE_MM:.0f} mm side)")
print("Print at 100 % scale, no fit-to-page, then measure the black square to")
print("confirm the actual printed size and use that value in frames.py.")