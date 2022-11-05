import re
import pandas as pd
from DatasetTrasformation import *
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
data = pd.DataFrame(pd.read_json('train.json/train.json'))

print(f"\nColonne del Dataset: {data.columns}")
print(f"\n{data.info()}\n")

print(f"Occorrenza di ogni etichetta\n{data['cuisine'].value_counts()}\n")# ->Dataset Sbilanciato
print(f"#ricette: {data['cuisine'].value_counts().sum()}\n")# ->Dataset Sbilanciato
print(f"#etichette : {data['cuisine'].value_counts().count()}") # -> 20 Classi differenti


#################################### INIZIO EDA ####################################
ingredients = ingredientsToColumns(data)
print("\n#ricette " + str(len(ingredients))) # -> 39774 ricette


#converto da lista di ingredienti a Series(ingredienti)
ingredients = ingredients.apply(pd.Series).stack().reset_index(drop = True)  # Dimensione, 428275 ingredienti con ripetizioni


#calcolo le occorrenze di ogni ingrediente
index = pd.Series(ingredients).drop_duplicates().sort_values(axis = 0, ascending=True)
valueCount_ing = pd.Series(ingredients.value_counts(), index = index)

print(f'\nIngredienti e rispettive occorrenze \n{valueCount_ing}')
print(f'\nIngrediente con più ripetizioni :{valueCount_ing.values.max()}')
print(f'\nSomma di tutte le occerrenze : {valueCount_ing.sum(axis=0)}')

# print(valueCount_ing) #Molti ingredienti diversi -> 6668 ingredienti

# Osservo la frequanza degli ingredienti
bin = [1, 10, 20, 50, 100, 200, 400, 800, 1000, 2000, 3000, 6000, 18049]
labels = []
for i in range(len(bin)-1):
    labels.append(str(bin[i]) + "-" + str(bin[i+1]))

valbin =pd.cut(valueCount_ing.values, bin, labels=labels, include_lowest=True)
counts = pd.value_counts(valbin)

fig, ax = plt.subplots()
ax.bar(labels, counts)
plt.show()

