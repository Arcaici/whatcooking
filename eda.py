#EDA file
#This file contains technique of eploratory data analysis
#for understand how data inside train.json file are made

#Dataset quick sheet
#
#Problem type       : Classification problem
#number of samples  : 39774
#label              : cuisine
#type               : text
#unique labels      : 20
#Dataset balance    : unbalanced
#
#total ingredients  : 428275
#unique ingredients :

import re
import pandas as pd
from featureExctraction_Cleaning_Encoding import *
import matplotlib.pyplot as plt

#setting pandas options
pd.set_option("display.max_columns", None)

#loading data
data = pd.DataFrame(pd.read_json('train.json/train.json'))

#general information
print(f"\nData Column: {data.columns}\n")
print(f"\n{data.info()}\n")
print(f"\n{data.head(10)}\n")

#understanding dataset's balance and number of samples
print(f"labels occurency\n{data['cuisine'].value_counts()}\n")
print(f"total recipes: {data['cuisine'].value_counts().sum()}\n")
print(f"unique labels : {data['cuisine'].value_counts().count()}")

#cleaning ingredients (see Cleaning data)
ingredients = ingredientsExtraction(data, 'only')

#list of all ingredients with repetead ones
ingredients = ingredients.apply(pd.Series).stack().reset_index(drop = True)
print(ingredients)

#ingredients occurrency
ing_unique_index = pd.Series(ingredients).drop_duplicates().sort_values(axis = 0, ascending=True)
ingredients_occurrency = pd.Series(ingredients.value_counts(), index = ing_unique_index)

print(f'\ningredients occurrency\n{ingredients_occurrency}')
print(f'\ningredients with max occurrency :{ingredients_occurrency.values.max()}')

#plotting a bar graph with ingredients occurrency
bin = [1, 10, 20, 50, 100, 200, 400, 800, 1000, 2000, 3000, 6000, 18049]
labels = []
for i in range(len(bin)-1):
    labels.append(str(bin[i]) + "-" + str(bin[i+1]))

valbin =pd.cut(ingredients_occurrency.values, bin, labels=labels, include_lowest=True)
counts = pd.value_counts(valbin)


px = 1/plt.rcParams['figure.dpi']  # pixel in inches

fig, ax = plt.subplots(figsize=(1400*px, 800*px))

ax.bar(labels, counts,color = (0,0,1, 0.6))

ax.set_title("Ingredients occurrency")
ax.set_ylabel("number of features ")
ax.set_xlabel("ingredients occurrencys bin")

plt.show()

fig, ax = plt.subplots(figsize=(1400*px, 800*px))

ax.bar(labels, np.log(counts),color = (1,0,0, 0.6))

ax.set_title("Ingredients occurrency")
ax.set_ylabel("number of features (log scale)")
ax.set_xlabel("ingredients occurrencys bin")

plt.show()

#conclusion

    #Features
    #Features are extracted from the list ingredients from each sample, the total ammount of unique features is 6645,
    #and i will use just some of them because a lot of them are present just few times over all recipes

    #Dataset
    # the dataset is strictly ambalaced so i will use f1-score as measurament for understand how the model generalize well the uknowown problem.