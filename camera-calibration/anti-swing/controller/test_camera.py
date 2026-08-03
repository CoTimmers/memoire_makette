"""Camera-only test: detect the marker and locate the attachment point of the crate.

The computed point is projected back onto the image as a red circle. If it does
not land where the cable actually attaches, adjust MARKER_SIZE or COM_OFFSET.

Keys:
    w / s   offset along the marker Y axis  (green)  +/- 1 cm
    a / d   offset along the marker X axis  (red)    +/- 1 cm
    e / c   offset along the marker Z axis  (blue)   +/- 1 cm
    p       print the current offset
    q       quit
"""

import cv2
import numpy as np
import pickle
import os
import time

CAMERA_ID   = 0
MARKER_SIZE = 0.157                       # printed marker side [m] - measure it
COM_OFFSET  = np.array([0.0, 0.13, 0.0])  # marker -> attachment point [m]
CALIB_FILE  = "output/calibration_data.pkl"

# ---------------- camera intrinsics ----------------
if os.path.exists(CALIB_FILE):
    with open(CALIB_FILE, "rb") as f:
        data = pickle.load(f)
    mtx = np.array(data.get("camera_matrix", data.get("mtx")))
    dist = np.array(data.get("distortion_coefficients", data.get("dist")))
    calibrated = True
else:
    mtx, dist, calibrated = None, np.zeros(5), False
    print("No calibration file: using a rough guess, distances will be approximate.")

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters())

h = MARKER_SIZE / 2
obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
print(f"resolution {int(cap.get(3))}x{int(cap.get(4))}")

offset = COM_OFFSET.astype(float).copy()
t_prev, dt_list = None, []

while True:
    ok, frame = cap.read()
    if not ok:
        break
    t_now = time.perf_counter()
    if t_prev is not None:
        dt_list.append(t_now - t_prev)
        if len(dt_list) > 120:
            dt_list.pop(0)
    t_prev = t_now
    H, W = frame.shape[:2]
    if mtx is None:                       # rough intrinsics if no calibration
        mtx = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=float)

    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    fps_txt = (f"{1/np.mean(dt_list):.1f} fps measured (worst gap "
               f"{1000*max(dt_list):.0f} ms)") if dt_list else "..."
    lines = [f"{W}x{H}   {fps_txt}",
             f"marker size {MARKER_SIZE*100:.1f} cm   "
             f"offset [{offset[0]*100:+.0f}, {offset[1]*100:+.0f}, {offset[2]*100:+.0f}] cm",
             "calibrated" if calibrated else "NOT calibrated: distances approximate"]

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        ok, rvec, tvec = cv2.solvePnP(obj, corners[0][0], mtx, dist)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            com = tvec.flatten() + R @ offset

            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, MARKER_SIZE * 0.5)

            # project the computed point back onto the image
            pt, _ = cv2.projectPoints(com.reshape(1, 3), np.zeros(3), np.zeros(3), mtx, dist)
            u, v = int(pt[0][0][0]), int(pt[0][0][1])
            cv2.circle(frame, (u, v), 9, (0, 0, 255), 2)
            cv2.circle(frame, (u, v), 2, (0, 0, 255), -1)
            cv2.putText(frame, "attachment point", (u + 14, v + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            m = tvec.flatten()
            lines += [f"marker  x {m[0]:+.3f}  y {m[1]:+.3f}  z {m[2]:+.3f} m",
                      f"point   x {com[0]:+.3f}  y {com[1]:+.3f}  z {com[2]:+.3f} m",
                      f"distance to marker {np.linalg.norm(m):.3f} m"]
    else:
        lines.append("no marker detected")

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("camera test", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k == ord("w"):
        offset[1] += 0.01
    elif k == ord("s"):
        offset[1] -= 0.01
    elif k == ord("d"):
        offset[0] += 0.01
    elif k == ord("a"):
        offset[0] -= 0.01
    elif k == ord("e"):
        offset[2] += 0.01
    elif k == ord("c"):
        offset[2] -= 0.01
    elif k == ord("p"):
        print(f"COM_OFFSET = np.array([{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}])")

cap.release()
cv2.destroyAllWindows()
print(f"final offset: COM_OFFSET = np.array([{offset[0]:.3f}, "
      f"{offset[1]:.3f}, {offset[2]:.3f}])")