# memoire_makette

Explication scientifique
Montre le code des forces idéales
Demander pour faire le teste en réel parce que je dois prendre en compte les erreurs des inputs de la grue (donc considérer que le système n'est jamais parfait)

Design une autre facon de parois icnlinées pour empiler le bac. 




%%%%%%%%%%%%%%%%%

Trouver la range de la trajectoire possible de la grue pour que le bac suive la position du mur qui pivote. 
Attention si la grue s'éloigne trop, le bac va pivoter. Donc il faut mettre une limite de position. 

On impose toujours une vitesse constante au mur qui pivote. 
--> On veut connaitre la direction de traction du cable à chaque time-step --> Donner une limite en fonction de l'orientation du mur. 

* (Est-ce qu'on veut que ce soit faire le plus rapidement possible ?) Oui on va pas attendre que le mur ait fait 90° pour changer de commande. 
*  
