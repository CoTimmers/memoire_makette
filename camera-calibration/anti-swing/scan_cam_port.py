"""Find which camera index and backend actually work with OpenCV."""

import cv2

backends = [(cv2.CAP_DSHOW, "CAP_DSHOW"),
            (cv2.CAP_MSMF, "CAP_MSMF"),
            (cv2.CAP_ANY, "CAP_ANY")]

trouve = False
for backend, nom in backends:
    for i in range(4):
        cap = cv2.VideoCapture(i, backend)
        ok, frame = cap.read()
        if ok and frame is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"OK   index {i}   {nom:10s}   "
                  f"{frame.shape[1]}x{frame.shape[0]}   {fps:.0f} fps")
            trouve = True
        cap.release()

if not trouve:
    print("Aucune camera accessible. Fermer Teams/Zoom/l'appli Camera, "
          "verifier les parametres de confidentialite Windows, essayer un autre port USB.")