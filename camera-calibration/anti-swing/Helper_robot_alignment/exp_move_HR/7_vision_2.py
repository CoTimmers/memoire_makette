"""Vision pour l'experience 7: alignement du helper sur le bac.

    ID_CHARGE = 12   marqueur du helper, suspendu au cable
    ID_REF    =  8   marqueur du bac, fixe, definit le repere monde

Lancable seul pour verifier la geometrie, sans robot:

    python 7_vision_2.py

La fenetre s'ouvre, on tourne le helper a la main, et on voit D1 et D3 se
deplacer avec phi. C'est le moyen le plus rapide de valider la formule et son
signe avant de laisser le robot bouger.

Les trois points de travail
---------------------------
Tous en axes monde, origine sur le bac.

    D1   mise en place, en recul          x = X_D1        constant
                                          y = D_MUR cos(phi - 90) + MARGE_Y
    D3   arret net, tout pres du bac      x = X_D3        constant
                                          y = le meme que D1
    D4   repli fixe apres le choc

D_MUR est la distance du pivot au bout du mur qui va frapper. Comme
cos(phi - 90) = sin(phi), y suit la projection de ce mur sur l'axe y du bac:
le bout du mur se presente ainsi en face du bac quand on avance selon -x.
MARGE_Y ecarte de quelques centimetres pour que rien ne touche avant l'arret.

D1 et D3 ayant le meme y, l'approche est une ligne droite selon -x.

Qui calcule D1 et D3
--------------------
SUIVI_PHI decide:

    True    le module les recalcule a chaque image depuis phi. C'est le mode
            du test autonome, et celui du main pendant la mise en place.
    False   ils sont figes a leur derniere valeur. Le main bascule dessus au
            moment de construire la trajectoire, pour que la cible ne bouge
            plus sous les pieds du plan.

D4 n'est jamais recalcule.

Angle entre les deux reperes
----------------------------
    yaw   angle de l'axe x du helper dans le repere du bac        [rad]
    phi   = -yaw, la convention utilisee par le main et affichee  [rad]

Les deux sont publies. yaw reste pour les scripts qui l'utilisaient deja; phi
est celui dont le signe correspond au sens de correction voulu.

Dictionnaire partage, mis a jour par le thread
----------------------------------------------
    err_d1, err_d3, err_d4  [ex, ey]  pivot du helper -> le point, axes base [m]
    erreur      [ex, ey]     celle que CIBLE_ACTIVE designe                 [m]
    pos_monde   [x, y]       pivot en axes monde, origine au bac            [m]
    dist_origine float       norme de pos_monde                             [m]
    theta       [thx, thy]   angles de ballant bruts, axes base           [rad]
    yaw, phi    float        rotation du helper autour de la verticale    [rad]
    vus         (ref, charge)
    pret        bool         suspension calibree et une erreur calculee
    t           float        instant de PRISE de l'image (retard soustrait)
    l_mes       float        distance suspension -> point d'accroche        [m]

Deux points portent le nom de pivot dans cette experience, le vocabulaire est
donc strict:

    suspension     ou le cable rejoint l'outil. Fixe dans le repere camera,
                   trouve une fois par calibration, helper immobile.
    pivot_charge   ou le cable rejoint le helper, a OFFSET_PIVOT du marqueur 12.
                   C'est le point que la commande deplace, et celui dont la
                   distance aux points de travail constitue l'erreur.
"""

import cv2
import numpy as np
import pickle
import threading
import time

# ---------------- configuration ----------------
CAMERA_ID  = 0
CALIB_FILE = "output/calibration_data.pkl"

ID_CHARGE     = 12              # helper suspendu
ID_REF        = 8               # bac, reference monde
TAILLE_CHARGE = 0.100           # cote imprime du marqueur 12 [m]
TAILLE_REF    = 0.157           # cote imprime du marqueur 8  [m]

# Offsets dans le repere propre de chaque marqueur: ils tournent avec lui.
OFFSET_PIVOT   = np.array([0.25, 0.07, 0.0])    # marqueur 12 -> accroche cable
OFFSET_ORIGINE = np.array([0.20, -0.10, 0.0])   # marqueur 8  -> origine monde

# --- geometrie de D1 et D3 ---
D_MUR   = 0.30          # pivot -> bout du mur qui frappe [m]
MARGE_Y = 0.02          # ecart lateral avant l'arret net [m]
X_D1    = 0.30          # recul de la mise en place [m]
X_D3    = -0.1          # arret net, tout pres du bac [m]
SUIVI_PHI = True        # False pour figer D1 et D3 a leur valeur courante

# Points de travail, axes monde, origine au bac. D1 et D3 sont recalcules a
# chaque image tant que SUIVI_PHI est vrai; les valeurs ci-dessous ne servent
# qu'avant la premiere mesure de phi.
D1 = np.array([X_D1, MARGE_Y, 0.0])
D3 = np.array([X_D3, MARGE_Y, 0.0])
D4 = np.array([0.2828, -0.2828, 0.0])   # repli fixe apres le choc

RAYON_OK = 0.030                # rayon dans lequel un point compte comme atteint

# Point que la cle "erreur" suit. Le main ecrit dedans: "d1", "d3" ou "d4".
CIBLE_ACTIVE = "d1"

# Axes horizontaux camera -> axes base robot. De calib_cam2base.py.
CAM2BASE = np.array([[1.0, 0.0],
                     [0.0, -1.0]])

RETARD_CAM = 0.031              # prise -> disponibilite [s], de mesure_retard.py
N_CALIB    = 60                 # images pour localiser la suspension du cable
ALPHA_REF  = 0.5                # passe-bas sur le repere monde

AFFICHAGE = False               # True pour ouvrir la fenetre de controle

_stop = threading.Event()


def points(phi):
    """D1 et D3 en axes monde, origine au bac, pour l'angle phi.

    x est constant: l'approche est une ligne droite selon -x, de X_D1 a X_D3.
    y suit la projection du mur, cos(phi - 90) = sin(phi), plus la marge qui
    evite tout contact avant l'arret net.
    """
    y_mur = -D_MUR * np.cos(phi - np.pi / 2)
    y = y_mur + MARGE_Y * np.sign(y_mur)
    return (np.array([X_D1, y, 0.0]),
            np.array([X_D3, y, 0.0]))


def _projette(p, mtx, dist):
    pt, _ = cv2.projectPoints(np.asarray(p, dtype=np.float64).reshape(1, 3),
                              np.zeros(3), np.zeros(3), mtx, dist)
    return int(pt[0][0][0]), int(pt[0][0][1])


def _obj_points(taille):
    h = taille / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                    dtype=np.float32)


def _pose(corners, ids, cible_id, taille, mtx, dist):
    if ids is None:
        return None
    idx = np.where(ids.flatten() == cible_id)[0]
    if len(idx) == 0:
        return None
    ok, rvec, tvec = cv2.solvePnP(_obj_points(taille), corners[idx[0]][0],
                                  mtx, dist)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return tvec.flatten(), R


def _triede(frame, origine, R, mtx, dist, longueur=0.06):
    """Trace un repere 3D: X rouge, Y vert, Z bleu. Rend le pixel de l'origine."""
    o = _projette(origine, mtx, dist)
    for vec, couleur, nom in [((longueur, 0, 0), (0, 0, 255), "X"),
                              ((0, longueur, 0), (0, 255, 0), "Y"),
                              ((0, 0, longueur), (255, 0, 0), "Z")]:
        p = _projette(origine + R @ np.array(vec), mtx, dist)
        cv2.line(frame, o, p, couleur, 2, cv2.LINE_AA)
        cv2.putText(frame, nom, p, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    couleur, 1, cv2.LINE_AA)
    return o


def _texte(frame, s, org, echelle=0.6, couleur=(0, 255, 0), ep=1):
    """Texte lisible sur n'importe quel fond: contour noir puis remplissage."""
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, echelle,
                (0, 0, 0), ep + 3, cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, echelle,
                couleur, ep, cv2.LINE_AA)


def _loop(etat, l_cable):
    global D1, D3

    with open(CALIB_FILE, "rb") as f:
        data = pickle.load(f)
    mtx = np.array(data.get("camera_matrix", data.get("mtx")))
    dist = np.array(data.get("distortion_coefficients", data.get("dist")))

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    suspension, calib = None, []
    o_f, R_ref_f = None, None           # repere monde lisse

    while not _stop.is_set():
        ok, frame = cap.read()
        # cap.read() bloque jusqu'a disponibilite de l'image: l'horodatage est
        # pris ici et le retard du pipeline soustrait une seule fois.
        t = time.perf_counter() - RETARD_CAM
        if not ok:
            continue

        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        charge = _pose(corners, ids, ID_CHARGE, TAILLE_CHARGE, mtx, dist)
        ref = _pose(corners, ids, ID_REF, TAILLE_REF, mtx, dist)
        etat["vus"] = (ref is not None, charge is not None)

        if charge is None:
            continue
        p_c, R_c = charge
        pivot_charge = p_c + R_c @ OFFSET_PIVOT

        # ---- calibration de la suspension: helper immobile, cable vertical ----
        if suspension is None:
            calib.append(pivot_charge)
            if len(calib) >= N_CALIB:
                suspension = np.mean(calib, axis=0) - np.array([0.0, 0.0, l_cable])
                print(f"\nsuspension calibree: {np.round(suspension, 4)}")
            continue

        # ---- angles de ballant bruts ----
        c = pivot_charge - suspension
        th = np.array([np.arctan2(c[0], c[2]), np.arctan2(c[1], c[2])])
        etat["theta"] = CAM2BASE @ th
        etat["l_mes"] = float(np.linalg.norm(c))

        # ---- repere monde, filtre passe-bas ----
        # Le bac est fixe: lisser fort ne coute rien et retire le tremblement
        # d'orientation qu'un bras de levier de 40 cm amplifierait a D4.
        if ref is not None:
            p_r, R_r = ref
            o_brut = p_r + R_r @ OFFSET_ORIGINE
            if o_f is None:
                o_f, R_ref_f = o_brut, R_r
            else:
                o_f = (1 - ALPHA_REF) * o_f + ALPHA_REF * o_brut
                R_ref_f = (1 - ALPHA_REF) * R_ref_f + ALPHA_REF * R_r
                U, _, Vt = np.linalg.svd(R_ref_f)       # retour sur SO(3)
                R_ref_f = U @ Vt

        # ---- erreurs vers les trois points ----
        pts = {}
        if o_f is not None:
            o, R_ref = o_f, R_ref_f

            # Angle du helper: axe x du marqueur projete dans le plan monde.
            # Calcule avant les points, puisqu'ils en dependent.
            x_c = R_ref.T @ R_c[:, 0]
            yaw = float(np.arctan2(x_c[1], x_c[0]))
            etat["yaw"] = yaw
            etat["phi"] = -yaw          # convention de correction du main

            if SUIVI_PHI:
                D1, D3 = points(etat["phi"])

            for nom, D in (("d1", D1), ("d3", D3), ("d4", D4)):
                p = o + R_ref @ np.asarray(D, float)
                pts[nom] = p
                etat["err_" + nom] = CAM2BASE @ (p - pivot_charge)[:2]
            etat["erreur"] = etat.get("err_" + CIBLE_ACTIVE, etat["err_d1"])

            # position du helper en axes monde, bac a l'origine
            pos = (R_ref.T @ (pivot_charge - o))[:2]
            etat["pos_monde"] = pos
            etat["dist_origine"] = float(np.linalg.norm(pos))
            etat["a_d4"] = bool(np.linalg.norm(etat["err_d4"]) < RAYON_OK)

        etat["t"] = t
        etat["pret"] = "err_d3" in etat

        # ---- fenetre de controle ----
        if AFFICHAGE:
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            u_p = _triede(frame, pivot_charge, R_c, mtx, dist)
            cv2.circle(frame, u_p, 6, (0, 165, 255), 2)
            _texte(frame, "pivot helper", (u_p[0] + 14, u_p[1] - 10), 0.5,
                   (0, 165, 255))

            if pts:
                u_o = _triede(frame, o, R_ref, mtx, dist)
                cv2.circle(frame, u_o, 6, (255, 255, 0), 2)
                _texte(frame, "origine (bac)", (u_o[0] + 14, u_o[1] - 10), 0.5,
                       (255, 255, 0))

                for nom in ("d4", "d1", "d3"):
                    u = _projette(pts[nom], mtx, dist)
                    actif = (nom == CIBLE_ACTIVE)
                    couleur = (0, 200, 0) if actif else (200, 200, 200)
                    cv2.drawMarker(frame, u, couleur, cv2.MARKER_TILTED_CROSS,
                                   16, 2 if actif else 1)
                    cv2.circle(frame, u, 5, couleur, 1)
                    _texte(frame, nom.upper(), (u[0] + 10, u[1] + 4), 0.6,
                           couleur, 2)

                u_act = _projette(pts.get(CIBLE_ACTIVE, pts["d1"]), mtx, dist)
                cv2.arrowedLine(frame, u_p, u_act, (0, 255, 255), 1,
                                cv2.LINE_AA, tipLength=0.05)
                # la ligne D1 -> D3 montre le chemin de l'approche
                cv2.line(frame, _projette(pts["d1"], mtx, dist),
                         _projette(pts["d3"], mtx, dist), (120, 120, 120), 1,
                         cv2.LINE_AA)

            e = etat.get("erreur", np.zeros(2))
            pos = etat.get("pos_monde", np.zeros(2))
            lignes = [f"cible active {CIBLE_ACTIVE.upper()}   "
                      f"error {1000*e[0]:+6.0f}, {1000*e[1]:+6.0f} mm  "
                      f"|e| {1000*np.linalg.norm(e):5.0f} mm",
                      f"helper/monde {1000*pos[0]:+6.0f}, {1000*pos[1]:+6.0f} mm  "
                      f"dist origine {1000*etat.get('dist_origine', 0):5.0f} mm",
                      f"D1 {1000*D1[0]:+6.0f},{1000*D1[1]:+6.0f}   "
                      f"D3 {1000*D3[0]:+6.0f},{1000*D3[1]:+6.0f}   "
                      f"D4 {1000*D4[0]:+6.0f},{1000*D4[1]:+6.0f} mm"
                      f"   {'(suit phi)' if SUIVI_PHI else '(figes)'}",
                      f"theta        {np.degrees(etat['theta'][0]):+5.1f}, "
                      f"{np.degrees(etat['theta'][1]):+5.1f} deg",
                      f"l_mes        {etat['l_mes']:.3f} m  (attendu "
                      f"{l_cable:.3f})",
                      f"vus          ref(8) {ref is not None}   "
                      f"helper(12) True"]
            for i, s in enumerate(lignes):
                _texte(frame, s, (10, 28 + 26 * i))

            # phi en gros, c'est la grandeur que l'essai cherche a annuler
            _texte(frame, f"phi = {np.degrees(etat.get('phi', 0.0)):+.1f} deg",
                   (10, 28 + 26 * len(lignes) + 30), 1.3, (0, 255, 0), 2)

            cv2.imshow("vision 7", frame)
            cv2.waitKey(1)

    cap.release()
    if AFFICHAGE:
        cv2.destroyAllWindows()


def start(etat, l_cable):
    etat.update({"theta": np.zeros(2), "pos_monde": np.zeros(2),
                 "yaw": 0.0, "phi": 0.0, "dist_origine": 0.0, "a_d4": False,
                 "vus": (False, False), "pret": False,
                 "t": time.perf_counter(), "l_mes": l_cable})
    threading.Thread(target=_loop, args=(etat, l_cable), daemon=True).start()


def stop():
    _stop.set()


# ---------------------------------------------------------------- test seul
if __name__ == "__main__":
    L_CABLE = 1.0

    AFFICHAGE = True
    etat = {}
    start(etat, L_CABLE)
    print(f"D_MUR {1000*D_MUR:.0f} mm   MARGE_Y {1000*MARGE_Y:.0f} mm   "
          f"X_D1 {1000*X_D1:.0f} mm   X_D3 {1000*X_D3:.0f} mm")
    print("helper immobile pour la calibration de la suspension...")
    while not etat["pret"]:
        time.sleep(0.1)
    print("pret. Tourner le helper a la main pour voir D1 et D3 suivre phi.")
    print("Ctrl-C pour arreter.\n")

    try:
        while True:
            print(f"\rphi {np.degrees(etat['phi']):+7.1f} deg | "
                  f"D1 {1000*D1[0]:+5.0f},{1000*D1[1]:+5.0f} | "
                  f"D3 {1000*D3[0]:+5.0f},{1000*D3[1]:+5.0f} mm | "
                  f"|e| {1000*np.linalg.norm(etat['erreur']):5.0f} mm | "
                  f"l_mes {etat['l_mes']:.3f} m | "
                  f"vus {int(etat['vus'][0])}{int(etat['vus'][1])}",
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stop()
        time.sleep(0.3)
        print("\narrete.")