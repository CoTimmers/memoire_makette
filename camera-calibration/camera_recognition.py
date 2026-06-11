import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

# cap = cv2.VideoCapture(1)

# ret, frame = cap.read()

# print(ret)

# print(frame)

# plt.imshow(frame)

# cap.release()

# def take_photo():
#     cap = cv2.VideoCapture(1)
#     ret, frame = cap.read()
#     cv2.imwrite('photo.jpg', frame)
#     cap.release()

# take_photo()




#Rendering in real time 
# cap = cv2.VideoCapture(1)
# while cap.isOpened():
#     ret, frame = cap.read()

#     # sshoww imagess
#     cv2.imshow('webcam', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


#Premier test de reconnaissance de marqueurs ArUco

# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# aruco_params = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# cap = cv2.VideoCapture(1)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Détecter les marqueurs
#     corners, ids, rejected = detector.detectMarkers(frame)

#     if ids is not None:
#         # Dessiner les marqueurs détectés
#         cv2.aruco.drawDetectedMarkers(frame, corners, ids)
#         print(f"Marqueur détecté ! ID: {ids[0][0]}")

#     cv2.imshow('ArUco Tracking', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



################


# cap = cv2.VideoCapture(1)
# photo_count = 0

# print("Appuie sur ESPACE pour prendre une photo, Q pour quitter")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     cv2.putText(frame, f"Photos: {photo_count}/20 - ESPACE=photo Q=quitter",
#                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     cv2.imshow('Calibration', frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord(' '):  # espace = photo
#         filename = f'calib_{photo_count:02d}.jpg'
#         cv2.imwrite(filename, frame)
#         photo_count += 1
#         print(f"Photo {photo_count} prise !")
#     elif key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





# import cv2

# cap = cv2.VideoCapture(1)
# board_size = (3, 6)  # même valeur que dans l'étape 3
# photo_count = 0

# print("Appuie sur ESPACE pour prendre une photo, Q pour quitter")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     found, corners = cv2.findChessboardCorners(gray, board_size, None)

#     if found:
#         # Dessine les coins détectés en vert
#         cv2.drawChessboardCorners(frame, board_size, corners, found)
#         cv2.putText(frame, "DAMIER DETECTE ! Appuie sur ESPACE",
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     else:
#         cv2.putText(frame, f"Damier non trouve... ({photo_count}/20)",
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     cv2.imshow('Calibration', frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord(' ') and found:  # photo seulement si damier détecté
#         filename = f'calib_{photo_count:02d}.jpg'
#         cv2.imwrite(filename, frame)
#         photo_count += 1
#         print(f"Photo {photo_count} prise !")
#     elif key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



#############

board_size = (5, 8)
square_size = 0.027  # 2.7cm par case

# Points 3D du damier (dans le monde réel)
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
objp *= square_size

obj_points = []  # points 3D
img_points = []  # points 2D dans l'image

# Charger toutes les photos de calibration
images = glob.glob('calib_*.jpg')
print(f"{len(images)} photos trouvées")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Chercher les coins du damier
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    if ret:
        obj_points.append(objp)
        # Affiner la précision des coins
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points.append(corners2)
        print(f" {fname} → damier trouvé")
    else:
        print(f" {fname} → damier non trouvé")

# Calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

# Sauvegarder les paramètres
np.save('camera_matrix.npy', camera_matrix)
np.save('dist_coeffs.npy', dist_coeffs)

print(f"\n Calibration terminée !")
print(f"Erreur de reprojection : {ret:.4f} (idéal < 0.5)")
print(f"Matrice caméra :\n{camera_matrix}")













# import cv2
# import numpy as np

# board_size = (9, 6)
# img = np.ones((700, 1000), dtype=np.uint8) * 255

# for i in range(board_size[1]):
#     for j in range(board_size[0]):
#         if (i + j) % 2 == 0:
#             x = j * 100 + 50
#             y = i * 100 + 50
#             img[y:y+100, x:x+100] = 0

# cv2.imwrite('checkerboard.png', img)
# print("Damier sauvegardé !")