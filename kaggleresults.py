import pandas as pd
from matplotlib import pyplot as plt

from featureExctraction_Cleaning_Encoding import *
import seaborn as sns; sns.set()
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier

pd.set_option("display.max_columns", None)

print('Loading data... \n')
train = pd.DataFrame(pd.read_json('train.json/train.json'))
test  = pd.DataFrame(pd.read_json('test.json/test.json'))

print('Cleaning data... \n')
train = pd.DataFrame(ingredientsExtraction(train, 'extraction'))
test  = pd.DataFrame(ingredientsExtraction(test, 'extraction'))

test = test.set_index('id')
print(test)
X_train    = pd.Series(train['ingredients'])
t_train    = pd.Series(train['cuisine'])
target_names = t_train.drop_duplicates().array

print('Encoding labels...\n')
t_train, le = encodeCusine(t_train)
t_ = pd.Series(t_train)

print('Splitting dataset (train, test)... \n')
clf = make_pipeline(TfidfVectorizer(min_df=50, max_df=6000, ngram_range=(1, 2)),
                    MLPClassifier(random_state=1, hidden_layer_sizes=(300,100), max_iter=300, solver="sgd",
                                  alpha= 0.0001))

print('Training... \n')
clf.fit(X_train, t_train)

print('Predict... \n')
result = pd.DataFrame(columns=['id','cuisine'])
test = test.reset_index()
id = test['id']
test = test['ingredients']

label = clf.predict(test)
label_decoded = decodeCusine(label, le)
for index in range(9944):
    result = result.append({'id': id[index], 'cuisine': label_decoded[index]}, ignore_index=True)

print('Saving results to csv file... \n')
result.to_csv('kaggle_results.csv', index=False)
