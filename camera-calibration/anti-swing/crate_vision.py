# -*- coding: utf-8 -*-
"""
Estime l'angle du cable (theta_x, theta_y) et ses derivees a partir d'un marqueur
ArUco fixe sur le bac, vu par une camera montee sur l'end-effector du UR10
(configuration "eye-in-hand", objectif pointant vers le bas).

Hypothese cle : l'orientation du TCP reste fixe pendant la manoeuvre anti-ballant.
L'axe Z de la camera reste donc aligne avec la verticale (gravite), et le vecteur
camera -> crochet est constant (camera et crochet sont solidaires du meme corps
rigide, l'end-effector). Ce vecteur ("pivot") est mesure une seule fois par
calibration : bac immobile, on moyenne la position du marqueur puis on retire
L_CABLE le long de l'axe Z. C'est la meme methode que l'ancien crate_state
(camera fixe), juste generalisee a 2 angles (X et Y) au lieu d'un seul.

CAM_YAW : rotation (rad) autour de la verticale entre les axes de la camera et
les axes de la base du robot. A 0 si le montage caméra est aligne avec l'axe X
du robot ; sinon calibrer ce parametre avant utilisation.
"""

import numpy as np
import cv2


class CrateVision:
    def __init__(self, camera_matrix, dist_coeffs, marker_size, com_offset,
                 l_cable, n_deriv=3, cam_yaw=0.0,
                 aruco_dict_id=cv2.aruco.DICT_4X4_50):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.com_offset = com_offset
        self.l_cable = l_cable
        self.cam_yaw = cam_yaw

        h = marker_size / 2
        self.obj_points = np.array(
            [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32
        )
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(aruco_dict_id),
            cv2.aruco.DetectorParameters(),
        )

        self.pivot = None                        # camera -> crochet, calibre une fois
        self._histo = []                          # (t, theta_x, theta_y) recents
        self._n_deriv = n_deriv

    # ------------------------------------------------------------------
    def _detect_com(self, frame):
        """Position du centre de masse du bac dans le repere camera, ou None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None
        ok, rvec, tvec = cv2.solvePnP(
            self.obj_points, corners[0][0], self.camera_matrix, self.dist_coeffs
        )
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return tvec.flatten() + R @ self.com_offset

    # ------------------------------------------------------------------
    def calibrate(self, cap, n_frames=60):
        """A appeler bac immobile : mesure le pivot (camera -> crochet)."""
        samples = []
        while len(samples) < n_frames:
            ret, frame = cap.read()
            if not ret:
                continue
            com = self._detect_com(frame)
            if com is not None:
                samples.append(com)
        moyenne = np.mean(samples, axis=0)
        self.pivot = moyenne - np.array([0.0, 0.0, self.l_cable])
        self._histo.clear()
        return self.pivot

    # ------------------------------------------------------------------
    def update(self, frame, t):
        """
        A appeler a chaque image.
        Renvoie (theta_x, theta_y, theta_x_dot, theta_y_dot, l_meas, ok).
        ok=False si le marqueur n'est pas detecte sur cette image.
        """
        if self.pivot is None:
            raise RuntimeError("calibrate() doit etre appele avant update()")

        com = self._detect_com(frame)
        if com is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, False

        c = com - self.pivot
        if self.cam_yaw != 0.0:
            ca, sa = np.cos(self.cam_yaw), np.sin(self.cam_yaw)
            cx, cy = c[0], c[1]
            c[0] = ca * cx - sa * cy
            c[1] = sa * cx + ca * cy

        theta_x = float(np.arctan2(c[0], c[2]))
        theta_y = float(np.arctan2(c[1], c[2]))
        l_meas = float(np.linalg.norm(c))

        self._histo.append((t, theta_x, theta_y))
        if len(self._histo) > self._n_deriv + 1:
            self._histo.pop(0)

        if len(self._histo) == self._n_deriv + 1:
            t0, tx0, ty0 = self._histo[0]
            dt = t - t0
            theta_x_dot = (theta_x - tx0) / dt
            theta_y_dot = (theta_y - ty0) / dt
        else:
            theta_x_dot = theta_y_dot = 0.0

        return theta_x, theta_y, theta_x_dot, theta_y_dot, l_meas, True


# ========================================================================
# Test autonome : affiche theta_x / theta_y en direct, juste avec la camera
# (pas besoin du robot ni d'ur_rtde). Bac immobile au demarrage pour calibrer.
# ========================================================================
if __name__ == "__main__":
    import pickle
    import time

    CAMERA_ID = 1
    MARKER_SIZE = 0.157
    CALIBRATION_FILE = "output/calibration_data.pkl"
    COM_OFFSET = np.array([0.0, 0.13, 0.0])
    L_CABLE = 0.60
    N_CALIB = 60

    with open(CALIBRATION_FILE, "rb") as f:
        data = pickle.load(f)
    camera_matrix = np.array(data.get("camera_matrix", data.get("mtx")))
    dist_coeffs = np.array(data.get("distortion_coefficients", data.get("dist")))

    vision = CrateVision(camera_matrix, dist_coeffs, MARKER_SIZE, COM_OFFSET, L_CABLE)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    print(f"Calibration du pivot sur {N_CALIB} images : BAC IMMOBILE svp...")
    pivot = vision.calibrate(cap, N_CALIB)
    print(f"[PIVOT] camera -> crochet : {np.round(pivot, 4)}")

    t_start = time.perf_counter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t = time.perf_counter() - t_start
            tx, ty, txd, tyd, l_meas, ok = vision.update(frame, t)

            if ok:
                txt = (f"tx {np.degrees(tx):+.1f} deg  ty {np.degrees(ty):+.1f} deg  "
                       f"txd {txd:+.2f}  tyd {tyd:+.2f}  l {l_meas:.3f} m")
                cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Etat du bac (eye-in-hand)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
