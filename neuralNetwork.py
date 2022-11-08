
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

print('\n Loading data... \n')

pd.set_option("display.max_columns", None)


data = pd.DataFrame(pd.read_json('train.json/train.json'))

print('\n Cleaning data... \n')


data = pd.DataFrame(ingredientsExtraction(data))

X = pd.Series(data['ingredients'])
t = pd.Series(data['cuisine'])
target_names = t.drop_duplicates().array

print('\n Encoding labels\n')
t, le = encodeCusine(t)
t = pd.Series(t)

print('\n Splitting dataset (train, validation, test)... \n')

X_train, X_temp, t_train, t_temp = train_test_split( X, t, test_size=0.2, random_state=42)
X_val, X_test, t_val, t_test = train_test_split( X, t, test_size=0.5, random_state=42)

# print('\n Model selection...\n')
#
#
# lrs = [0.0001, 0.001, 0.01, 0.1]
#
# best_score_avg = 0
# best_param = 0
# for lr in lrs:
#     clf = make_pipeline(TfidfVectorizer(min_df=50, max_df=6000, ngram_range=(1, 2)),
#                         MLPClassifier(random_state=1, hidden_layer_sizes=(300, 100), max_iter=300, solver="sgd",
#                                       alpha=lr))
#     clf.fit(X_train, t_train)
#     labels_val = clf.predict(X_val)
#     score_avg = f1_score(t_val, labels_val, average='weighted')
#     if score_avg > best_score_avg :
#         best_score_avg = score_avg
#         best_param = lr
#
# print(f'learning rate = {best_param}') # -> 0.0001
# print(f' with f1_weighted of = {best_score_avg}') #-> 0.80

best_param = 0.0001

print('\n Testing data on D_Test... \n')

X_train = pd.concat([X_train, X_val])
t_train = pd.concat([t_train, t_val])

clf = make_pipeline(TfidfVectorizer(min_df=50, max_df=6000, ngram_range=(1, 2)),
                    MLPClassifier(random_state=1, hidden_layer_sizes=(300, 100), max_iter=300, solver="sgd",
                                  alpha= best_param))

clf.fit(X_train, t_train)
labels = clf.predict(X_test)

print('\n Final Score \n')
print(classification_report(t_test, labels, target_names= target_names)) #f1_weighted -> 0.80



# neuralnet_score = classification_report(t_test, labels, target_names= target_names, output_dict= True)
# score = pd.DataFrame(neuralnet_score)
# score = score.T
# score.to_excel("neuralnet_results.xlsx")


t_test = decodeCusine(t_test, le)
labels = decodeCusine(labels, le)

mat = confusion_matrix(t_test, labels, labels= target_names)
sns.heatmap(mat.T, square = True, annot = True, fmt ='d', cbar = False, xticklabels = target_names, yticklabels = target_names)
plt.title('Confusion Matrix Neural Network')
plt.xlabel('true label')
plt.ylabel('predicted labels')
# plt.savefig('neuralNet_confusionMatrix', bbox_inches='tight')
plt.show()

