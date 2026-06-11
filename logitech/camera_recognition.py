import cv2
from matplotlib import pyplot as plt
import numpy as np

# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()


# cap.release()  # Important : libérer la caméra après la capture

# if ret:
#     # OpenCV lit en BGR, matplotlib affiche en RGB → il faut convertir
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     plt.imshow(frame_rgb)
#     plt.axis('off')
#     plt.show()
# else:
#     print("Impossible de lire la caméra")


# for i in range(5):
#     cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
#     if cap.isOpened():
#         print(f"Caméra trouvée à l'index {i}")
#         cap.release()
#     else:
#         print(f"Rien à l'index {i}")






# Choisir le dictionnaire ArUco
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# marker_id = 42
# marker_size = 200  # Size in pixels
# marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# cv2.imwrite('marker_42.png', marker_image)
# plt.imshow(marker_image, cmap='gray', interpolation='nearest')
# plt.axis('off')  # Hide axes
# plt.title(f'ArUco Marker {marker_id}')
# plt.show()




# Créer le damier
board_size = (9, 6)  # 9x6 cases intérieures
square_size = 0.025  # taille d'une case en mètres (2.5cm)

# Générer l'image du damier
img = np.ones((700, 1000), dtype=np.uint8) * 255

for i in range(board_size[1]):
    for j in range(board_size[0]):
        if (i + j) % 2 == 0:
            x = j * 100 + 50
            y = i * 100 + 50
            img[y:y+100, x:x+100] = 0

cv2.imwrite('checkerboard.png', img)
print("Damier sauvegardé !")