import json
from pypylon import pylon

from detect_green_marker import detect_marker
from move_crane import compute_command, send_command_to_crane, APPROACH_PIXEL, PIXELS_PER_METER
from check_alignment import is_aligned, is_aligned_wall2

with open("config/environment.json") as f:
    env = json.load(f)

# Paramètres
STEP_CM = 1
STEP_PX = int(0.01 * PIXELS_PER_METER)

# Wall 2 y pour le cornering
wall2_left  = env["wall_configurations"]["wall_closed"]["wall2"]["left"]
wall2_right = env["wall_configurations"]["wall_closed"]["wall2"]["right"]
WALL2_Y     = int((wall2_left[1] + wall2_right[1]) / 2)
OFFSET_3CM  = int(0.03 * PIXELS_PER_METER)
CORNERING_Y = WALL2_Y + OFFSET_3CM

# Connexion caméra Basler
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed

# State machine
state          = "APPROACH"
iteration_x    = 0   # itérations recul en x (approach)
iteration_y    = 0   # itérations avance en y (cornering)

print("=== FSM démarrée ===")
print(f"Approach point : {APPROACH_PIXEL} px")
print(f"Cornering y    : {CORNERING_Y} px (wall2_y={WALL2_Y} + 3cm)\n")

try:
    while camera.IsGrabbing():

        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            grab.Release()
            continue

        frame = converter.Convert(grab).GetArray()
        grab.Release()

        # ── APPROACH ──────────────────────────────────────────────
        if state == "APPROACH":
            current_target = (
                APPROACH_PIXEL[0] - (iteration_x * STEP_PX),
                APPROACH_PIXEL[1]
            )
            print(f"[APPROACH] iter={iteration_x} | cible={current_target}")

            result = detect_marker(frame)
            if result is None:
                print("  Marqueur non détecté, on attend...")
                continue

            com = result["pixel"]
            dx_m, dy_m = compute_command(com, current_target)
            print(f"  COM={com} | dx={dx_m*100:.1f}cm dy={dy_m*100:.1f}cm")
            send_command_to_crane(dx_m, dy_m)
            state = "CHECK_ALIGNED_WALL1"

        # ── CHECK ALIGNED WALL 1 ──────────────────────────────────
        elif state == "CHECK_ALIGNED_WALL1":
            aligned, d1, d2, side, side_cm = is_aligned(frame, draw=True)
            print(f"[CHECK_WALL1] d1={d1:.1f}cm d2={d2:.1f}cm "
                  f"côté={side_cm:.1f}cm ({side})")
            

            if aligned:
                if side == "LONG":
                    state = "FINAL_POSITIONING"
                else:
                    state = "CORNERING"
            else:
                state = "DECREASE_X"

        # ── DECREASE X ────────────────────────────────────────────
        elif state == "DECREASE_X":
            iteration_x += 1
            new_x = APPROACH_PIXEL[0] - (iteration_x * STEP_PX)
            print(f"[DECREASE_X] iter={iteration_x} | new x={new_x} px")
            state = "APPROACH"

        # ── CORNERING ─────────────────────────────────────────────
        elif state == "CORNERING":
            # x = même que la dernière itération de decrease_x
            # y = wall2_y + 3cm - iteration_y * 1cm
            cornering_target = (
                APPROACH_PIXEL[0] - (iteration_x * STEP_PX),
                CORNERING_Y + (iteration_y * STEP_PX)
            )
            print(f"[CORNERING] iter={iteration_y} | cible={cornering_target}")

            result = detect_marker(frame)
            if result is None:
                print("  Marqueur non détecté, on attend...")
                continue

            com = result["pixel"]
            dx_m, dy_m = compute_command(com, cornering_target)
            print(f"  COM={com} | dx={dx_m*100:.1f}cm dy={dy_m*100:.1f}cm")
            send_command_to_crane(dx_m, dy_m)
            state = "CHECK_ALIGNED_WALL2"

        # ── CHECK ALIGNED WALL 2 ──────────────────────────────────
        elif state == "CHECK_ALIGNED_WALL2":
            aligned, d1, d2 = is_aligned_wall2(frame, draw=True)
            print(f"[CHECK_WALL2] d1={d1:.1f}cm d2={d2:.1f}cm")

            if aligned:
                state = "PIVOTING"
            else:
                state = "INCREASE_Y"

        # ── INCREASE Y ────────────────────────────────────────────
        elif state == "INCREASE_Y":
            iteration_y += 1
            new_y = CORNERING_Y + (iteration_y * STEP_PX)
            print(f"[INCREASE_Y] iter={iteration_y} | new y={new_y} px")
            state = "CORNERING"

        # ── FINAL POSITIONING ─────────────────────────────────────
        elif state == "FINAL_POSITIONING":
            # x = même que la dernière itération de decrease_x
            # y = wall2_y + 8cm - iteration_y * 1cm
            final_target = (
                APPROACH_PIXEL[0] - (iteration_x * STEP_PX),
                WALL2_Y + int(0.08 * PIXELS_PER_METER) + (iteration_y * STEP_PX)
            )
            print(f"[FINAL_POSITIONING] iter={iteration_y} | cible={final_target}")

            result = detect_marker(frame)
            if result is None:
                print("  Marqueur non détecté, on attend...")
                continue

            com = result["pixel"]
            dx_m, dy_m = compute_command(com, final_target)
            print(f"  COM={com} | dx={dx_m*100:.1f}cm dy={dy_m*100:.1f}cm")
            send_command_to_crane(dx_m, dy_m)
            state = "CHECK_ALIGNED_WALL2_FINAL"

        # ── CHECK ALIGNED WALL 2 FINAL ────────────────────────────
        elif state == "CHECK_ALIGNED_WALL2_FINAL":
            aligned, d1, d2 = is_aligned_wall2(frame, draw=True)
            print(f"[CHECK_WALL2_FINAL] d1={d1:.1f}cm d2={d2:.1f}cm")

            if aligned:
                state = "PIVOTING"
            else:
                state = "INCREASE_Y_FINAL"

        # ── INCREASE Y FINAL ──────────────────────────────────────
        elif state == "INCREASE_Y_FINAL":
            iteration_y += 1
            new_y = WALL2_Y + int(0.08 * PIXELS_PER_METER) + (iteration_y * STEP_PX)
            print(f"[INCREASE_Y_FINAL] iter={iteration_y} | new y={new_y} px")
            state = "FINAL_POSITIONING"

        # ── PIVOTING ──────────────────────────────────────────────
        elif state == "PIVOTING":
            print("\n=== Côté court aligné wall2 → PIVOTING ===")
            # TODO: coder la phase pivoting
            break

except KeyboardInterrupt:
    print("\nArrêt manuel.")

finally:
    camera.StopGrabbing()
    camera.Close()
    print("Caméra déconnectée.")
