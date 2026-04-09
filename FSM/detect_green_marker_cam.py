import cv2
import numpy as np
import json
from pypylon import pylon

# Plage HSV calibrée sur la vraie caméra
LOWER_GREEN = np.array([44, 150, 100])
UPPER_GREEN = np.array([65, 255, 255])

with open("config/environment.json") as f:
    env = json.load(f)

METERS_PER_PIXEL = env["meters_per_pixel"]
WALL1_X_PIXEL = env["wall_configurations"]["wall_closed"]["wall1"]["top"][0]


def detect_marker(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    delta_x_m = (cx - WALL1_X_PIXEL) * METERS_PER_PIXEL

    return {
        "pixel": (cx, cy),
        "delta_x_m": round(delta_x_m, 4),
        "delta_x_cm": round(delta_x_m * 100, 2)
    }


if __name__ == "__main__":

    # Connexion à la caméra Basler GigE
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed

    print("Caméra connectée. Appuyez sur Q pour quitter.")

    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab.GrabSucceeded():
            image = converter.Convert(grab)
            frame = image.GetArray()

            result = detect_marker(frame)

            if result:
                cx, cy = result["pixel"]
                print(f"COM : {result['pixel']} | dx wall1 : {result['delta_x_cm']} cm")

                # Dessiner sur la frame
                cv2.circle(frame, (cx, cy), 12, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 20, (255, 255, 255), 3)
                cv2.putText(frame, f"COM ({cx},{cy})", (cx+24, cy-14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"dx={result['delta_x_cm']}cm", (cx+24, cy+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Marqueur non detecte", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow("Basler - Detection marqueur vert", frame)

        grab.Release()

        if cv2.waitKey(1) == ord('q'):
            break

    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
