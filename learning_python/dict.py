#dans les liste, nous pouvons avoir des repetions, cela veut dire deux fois le meme element
#dans les dictionaires, les cles doivent etre unique, cela veut dire quon peut pas avoir par exemple deux fois la cle nom dans un dictionaire
#personne par exemple 
myList = ["t", "phi", "xC", "yC", "th", "vx", "vy", "om", "xg", "yg"]
myList.append("john") #ajouter  un élément à la liste
myList.insert(1, "chris") 
myList.remove("chris")
print(myList.reverse())

# print(myList[0])
# print(myList[len(myList) - 1])


for item in myList:
    print(item)

# for i in range(len(myList)):
#     print(i, myList[i])


# numbers = [10, 20, 30, 40]

# for i, num in enumerate(numbers):
#     print(i, num)

# person = {
#     "name": "loic",
#     "age" : 25,
#     "city": "brussels"
# }

# print(person["city"])

# person["gender"] = "male" # ici nous faisons un insert
# person["city"] = "paris" # ici nous faisons un update

# for key in person:
#     print(key)
    
# for value in person.values():
#     print(value)

# for key, value in person.items():
#     print(key, value)
    
