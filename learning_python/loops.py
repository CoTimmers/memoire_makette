# for, on utilise un for quand nous savons en avance le nombre diterations a effectuer, eg si nous avons une liste chiffres et 
# que nous voulons trouver la somme des chiffres, du coup il faut iterer du debut jusqua la fin 

# while, une facon diterer en programmation, mais on lutilise quand nous savons pas en avance le nombre diterations a effecuter, eg
# si tu as une methode qui prend une liste de chiffres en parametre et que tu veux renvoyer lindice de la premiere chiffre negative
# cela veut dire que la chiffre peut se trouver au debut, au mileu, a la fin ou elle nest pas presente du tout dans la liste 
# du coup, tu utilises un while et tu tarretes des que tu trouves la chiffre et que tu evites les iterations qui ne sont pas necessaires 

chiffres = [1, 2, 3, 4]
somme = 0 
for k in chiffres:
    somme = somme + k
print(somme)
    
    
listeAvecChiffresNegatives = [1, 2, -3, 4]

    
def trouverIndiceChiffreNegative(liste):
    i = 0
    while (i < len(liste)):
        if liste[i] < 0:
            return i # si on trouve une chiffre negative, on renvoie la valeur et on arrete lexecution de la fonction
        i=i+1
    return -1 # si on arrive jusqua ici, cela veut dire quon a pas trouve des chiffres negative et que nous renvoyons une valeur que nous avons choisi comme eg -1 

print(trouverIndiceChiffreNegative(listeAvecChiffresNegatives))