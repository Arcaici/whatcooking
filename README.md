# Whatcooking 
> (First Machine Learning Project)

## Introduction
What's Cooking is a dataset available on [Kaggle](https://www.kaggle.com/competitions/whats-cooking) and provided by [yummly](https://www.yummly.com/about). The Dataset include a train.json file containing 39774 recipes with 20 different cusine types that results in a unbalanced dataset.

### Task
The challenge is to find a model that predict cusine types using the ingredients as features.

## Approches
In This project i decide to implement three differents models:
- *Naive Bayes*
- *Sofmax Regression*
- *Neaural Network*

The first two models were implemented using ***scikit-learn*** library, while Neural Network was implement with ***TensorFlow 2***.

### Features extraction
The dataset contained only two features: 
- _id_
- _ingredients_

Id was use just for identify predictions, while ingredients was processed for extract single ingredients for each recipe, and then i cleaned the ingredients by:
1. removing numbers
2. removing special chars
3. making every letters lowercase
4. removing extra whitespaces
5. removing units
6. applyng lemmatizer.

### libraries used
- *re* 
- *pandas*
- *scikit-learn.preprocessing*
- *ntlk.stem.WordNetLemmatizer*

## Exploratory Data Analisys
The eda results in a soft manner, checking the occurrency of each ingrient and plotting a bar chart that show how many ingredients have a range of n occurency, over differents range.

### libraries used
- *re* 
- *pandas*
- *matplotlib.pyplot*

### Charts
![Ingredients Occurency](https://github.com/Arcaici/whatcooking/blob/NeuralNetwork_wirh_TensowFlow/images/IngredientsOccurrency.png)
  
![Ingredients Occurency in Log Scale](https://github.com/Arcaici/whatcooking/blob/NeuralNetwork_wirh_TensowFlow/images/IngredientsOccurrency_logscale.png)

