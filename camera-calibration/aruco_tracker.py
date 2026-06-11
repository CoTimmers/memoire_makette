import cv2

# Choisir le même dictionnaire que dans le code de tracking
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Générer le marqueur (id=8, taille 400x400 pixels)
marker = cv2.aruco.generateImageMarker(aruco_dict, 7, 400)

# Sauvegarder
cv2.imwrite('aruco_marker.png', marker)
print("Marqueur sauvegardé !")