import pandas as pd
from featureExctraction_Cleaning_Encoding import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

print('Loading data... \n')
pd.set_option("display.max_columns", None)
data = pd.DataFrame(pd.read_json('train.json/train.json'))

print('Cleaning data... \n')
data = pd.DataFrame(ingredientsExtraction(data, 'extraction'))

X = pd.Series(data['ingredients'])
t = pd.Series(data['cuisine'])
target_names = t.drop_duplicates().array

print('Encoding labels...\n')
t, le = encodeCusine(t)
t = pd.Series(t)

print('Splitting dataset (train, validation, test)... \n')
X_train, X_temp, t_train, t_temp = train_test_split( X, t, test_size=0.2, random_state=42)
X_val, X_test, t_val, t_test = train_test_split( X_temp, t_temp, test_size=0.5, random_state=42)

il_neurons = 0