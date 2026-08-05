"""Genere un marqueur ArUco en PDF A4, a la taille physique exacte demandee.

    python genere_marqueur_pdf.py 12 100    -> marqueur id 12, cote 100 mm

Le PDF porte la taille en millimetres dans le fichier lui-meme : il suffit
d'imprimer a 100 % / "Taille reelle" (surtout PAS "Ajuster a la page").
Une reglette de controle de 100 mm est imprimee sous le marqueur.
"""

import sys
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image

ID = int(sys.argv[1]) if len(sys.argv) > 1 else 12
COTE_MM = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
DPI = 600  # resolution de l'image bitmap embarquee

dico = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
px = int(round(COTE_MM / 25.4 * DPI))
# arrondi au multiple de 6 (4x4 + bordure) pour des modules pixel-parfaits
px = max(6, (px // 6) * 6)

img = cv2.aruco.generateImageMarker(dico, ID, px)
pil = Image.fromarray(img)

nom = f"aruco_{ID}_{int(COTE_MM)}mm.pdf"
c = canvas.Canvas(nom, pagesize=A4)
W, H = A4

x = (W - COTE_MM * mm) / 2
y = H - 40 * mm - COTE_MM * mm

# marqueur, place a la taille physique exacte
c.drawImage(ImageReader(pil), x, y, width=COTE_MM * mm, height=COTE_MM * mm)

# reglette de controle 100 mm
ry = y - 20 * mm
c.setLineWidth(0.5)
c.line(x, ry, x + 100 * mm, ry)
for i in range(11):
    c.line(x + i * 10 * mm, ry, x + i * 10 * mm, ry + (4 * mm if i % 5 == 0 else 2 * mm))
c.setFont("Helvetica", 9)
c.drawString(x, ry - 5 * mm, "Reglette de controle : cette ligne doit mesurer exactement 100 mm")
c.drawString(x, ry - 10 * mm,
             f"ArUco DICT_4X4_50  id={ID}  cote nominal = {COTE_MM:.1f} mm")
c.drawString(x, ry - 15 * mm,
             "Imprimer a 100 % / Taille reelle -- PAS 'Ajuster a la page'.")

c.showPage()
c.save()
print(f"{nom} genere : cote {COTE_MM:.1f} mm, bitmap {px}x{px} px")
print("Mesurer le carre noir apres impression et reporter la valeur reelle dans frames.py")