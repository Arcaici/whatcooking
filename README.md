# Whatcooking 
> (First Machine Learning Project)

## Introduction
What's Cooking is a dataset available on [Kaggle](https://www.kaggle.com/competitions/whats-cooking) and provided by [yummly](https://www.yummly.com/about). The Dataset include a train.json file containing 39774 recipes with 20 different cuisine types that results in a unbalanced dataset.

### Task
The challenge is to find a model that predict cuisine types using the ingredients as features.

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

Id is use just for identify predictions, while ingredients is processed for extract single ingredients for each recipe, and then i clean the ingredients by:
1. removing numbers
2. removing special chars
3. making every letters lowercase
4. removing extra whitespaces
5. removing units
6. applyng lemmatizer.

### Libraries used
- *re* 
- *pandas*
- *scikit-learn.preprocessing*
- *ntlk.stem.WordNetLemmatizer*

## Exploratory Data Analisys
The eda results in a soft manner, checking the occurrency of each ingredient and plotting a bar chart that show how many ingredients have a range of n occurency, over differents ranges.

### Libraries used
- *re* 
- *pandas*
- *matplotlib.pyplot*

### Charts
![Ingredients Occurency](https://github.com/Arcaici/whatcooking/blob/NeuralNetwork_wirh_TensowFlow/images/IngredientsOccurrency.png)
  
![Ingredients Occurency in Log Scale](https://github.com/Arcaici/whatcooking/blob/NeuralNetwork_wirh_TensowFlow/images/IngredientsOccurrency_logscale.png)

## Model Tuning
Each model take in input words with a minimum document frequency of 50 units and a maximum document frequency of 6000 units, while for n-gram, 1-gram and 2-gram are choose as the best result. These statistics are choose using Naive Bayes, this model is used as a base model for check which words statistics perfom best over f1-micro-score, and they are use with the other models too.

### Performance metrics
I use f1-micro-score as performance metric, because the dataset task is not to better predict a class respect to the others, but to predict all classes with the same probability.

### Approaches
Each model is tune using grid search technique based on reguralization and learning-rate.

## Performance 
performance are calculated using f1-micro-score.

|result | score | regularisation | regularisation factor* | learning rate |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 1°  | 0.77 | n/a | n/a | 0.0001 |
| 1°  | 0.77  | l2 | 1.0 | n/a |
| 3°  | 0.71  | n/a | n/a | n/a |

*regularisation factor is 1.0 because the model doesn't use regularisation.

## Conclusion

The f1-micro-score that neural network and softmaxregression reached is good, but unfortunally the confusion matrix result with a lot of false positive, so the models do not perform well in all classes. All false positive predictions are check using _confusion matrix_ and individual f1-score are check too.  These results depend on the imbalance of the dataset, infact class with higher samples has higher f1-score.

### Possible implementation
There are two possible implementation:

* **Resampling:** for manage the imbalace of the dataset it could be a good practice to apply undersampling to large classes such as _italian_ and apply oversampling to smaller classes. It is not a one of the best practice, but sometimes can be really useful.
* **different models:** It could be interesting to try models like _Support Vector Machine_ or to check if there is a way to adapt this dataset to _Recurrent Neural Network_. 
