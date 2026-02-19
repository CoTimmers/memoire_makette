"je veux créer un code basic pour montrer c'est quoi le mouvement basic pour atteindre la position finale"
"La plaque rectangulaire commence collé au mur 1 et 2, avec le long coté  collé au mur 2"
"Mur 1: y=0"
"Mur 2: x=0"
"Mur 2 pivote de la position verticale, à vitesse constante pour atteindre + 90°"

"Première phase: Appliquer une force au COM pour rester coller au mur pendant le pivotement."
"Deuxième phase: Tirer le bac vers x+ pour que le coin supérieur gauche dépasse le pivot"
"Troisème phase: Fermer le mur qui pivote"
"Quatrième phase: Tirer le bac vers x- pour le positionner dans le coin"

"Je veux pouvoir faire tout le code moi même"

import numpy as np
import matplotlib.pyplot as plt


#Parameters

m = 7
a = 0.3
b = 0.4
I = (m/12.0) * (a**2 + b**2)
mu = 0.3

phi0 = 0
phiF = np.pi/2
T = 6
omega = (phiF - phi0)/T




pivot_world = np.array([0.0, 0.0], dtype=float)

#définir les points A & B

a_local = np.array([-a/2, -b/2], dtype=float)
b_local = np.array([-a/2, +b/2], dtype=float)

def rotation_matrix(theta):
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return r


def normal_vector(t):
    phi = phi0 + omega * t
    n1 = np.array([0.0, 1.0], dtype=float)
    n2 = np.array([np.cos(phi), np.sin(phi)], dtype=float)
    return n1, n2

def corners_position(xc, yc, psi) -> np.ndarray: # exprimer la position d'un point dans le monde en fonction du COM
    corners = np.array([
        [-a/2, -b/2], #A
        [-a/2,  b/2], #B
        [ a/2,  b/2], #C
        [ a/2, -b/2]  #D
    ], dtype=float)
    
    return (rotation_matrix(psi) @ corners.T).T + np.array([xc, yc], dtype=float)

print(corners_position(-0.15,-0.20, np.pi/2))

def distance(t, xc, yc, psi):
    n1, n2 = normal_vector(t)
    corners = corners_position(xc, yc, psi)
    da = corners[0]
    db = corners[1]
    dist_wall_2_a = np.dot(n2, da - pivot_world)
    dist_wall_2_b = np.dot(n2, db - pivot_world)
    return dist_wall_2_a, dist_wall_2_b

print(distance(6, 0.15, 0.20, np.pi/2))


def induced_force(f, state):
    x, y, psi, vx, vy, w = state
    Fc = np.array([-f, 0])
    orientation = rotation_matrix(psi) @ Fc
    return orientation

