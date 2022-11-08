
#intro

#naive bayes model is used for understand the unkwon model with an easy approch,
#this model can give some information about how i should tune the tf and id param
#for better extracting features from ingredients list, and which ngram are better performing in general.

#ngrams
#because we are talking about ingredients i assumed that it would be better to have at list 1 to 2 ngrams instead just 1,
# for these reason i choose to start the Grid search with (1,2) as n-gram.
#metrics
#i will use micro as a score metric, because it does favourite any class in particulari.
#in the previous version of all models i was using weighted average, it wasn't a good choise,
#because instead of evaluatting a model that perform well with all classes i was favouring the majority class (ex. italian).


import pandas as pd
import csv

from matplotlib import pyplot as plt

from featureExctraction_Cleaning_Encoding import *
import seaborn as sns; sns.set()
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB


pd.set_option("display.max_columns", None)

print('Loading data... \n')
data = pd.DataFrame(pd.read_json('train.json/train.json'))

print('Cleaning data... \n')
data = pd.DataFrame(ingredientsExtraction(data, "extraction"))

print(data.head())

X = pd.Series(data['ingredients'])
t = pd.Series(data['cuisine'])
target_names = t.drop_duplicates().array

print('Encoding labels\n')
t, le = encodeCusine(t)
t = pd.Series(t)


print('Splitting dataset (train, dev, test)... \n')

X_train, X_temp, t_train, t_temp = train_test_split( X, t, test_size=0.2, random_state=42)
X_val, X_test, t_val, t_test = train_test_split(X_temp, t_temp, test_size=0.5, random_state= 42)

print('Model selection... \n') # default values [ min_df = 1, max_df = 1.0, ngram_range 0 (1,1) ]

best_score_avg = 0

min_dfs = np.arange(10,310,20)
max_dfs = np.arange(2500,10100,500)
ngram_r = [(1, 2), (1, 3), (1, 4)]

for ngram in ngram_r:
    for min_df in min_dfs:
        for max_df in max_dfs:
            clf = make_pipeline(TfidfVectorizer(min_df=min_df, max_df= max_df, ngram_range=ngram), MultinomialNB())

            clf.fit(X_train, t_train)
            labels_val = clf.predict(X_val)
            score_avg = f1_score(t_val, labels_val, average='micro')

            print(f'min_df: {min_df}, max_df: {max_df}, ngram: {ngram}  f1_score_micro: {score_avg}')
            if score_avg > best_score_avg :
                best_score_avg = score_avg
                best_min_df = min_df
                best_max_df = max_df
                best_ngram_r = ngram

print(f'best min_df = {best_min_df}')               # min_df -> 50
print(f'best max_df = {best_max_df}')               # max_df -> 6000
print(f'best ngram_rg = {best_ngram_r}')            # ngram  -> (1,2)
print(f'with f1_micro of = {best_score_avg}')       # best   -> 0.71

best_min_df = 50
best_max_df = 6000
best_ngram_r= (1,2)

print('Testing data on D_test... \n')
tf = TfidfVectorizer(min_df=best_min_df, max_df= best_max_df, ngram_range=best_ngram_r)
X_shape = tf.fit_transform(X_train)
print(X_shape.shape)
clf = make_pipeline(TfidfVectorizer(min_df=50, max_df= 6000, ngram_range=(1,2)), MultinomialNB())

X_train = pd.concat([X_train, X_val])
t_train = pd.concat([t_train, t_val])

clf.fit(X_train, t_train)

print(X_train.shape)
print(X_train[0])
labels = clf.predict(X_test)

print('\n Final Score \n')
print(classification_report(t_test, labels, target_names= target_names)) # -> 0.70

# naive_bayes_score = classification_report(t_test, labels, target_names= target_names, output_dict= True)
# score = pd.DataFrame(naive_bayes_score)
# score = score.T
# score.to_excel("naiveBayes_results.xlsx")


t_test = decodeCusine(t_test, le)
labels = decodeCusine(labels, le)

mat = confusion_matrix(t_test, labels, labels=target_names)
sns.heatmap(mat.T, square = True, annot = True, fmt ='d', cbar = False , xticklabels = target_names, yticklabels = target_names)
plt.title('Confusion Matrix Naive Bayes')
plt.xlabel('true label')
plt.ylabel('predicted labels')
# plt.savefig('naiveBayes_confusionMatrix', bbox_inches='tight')
plt.show()



#conclusion
#the model selection results still is:
#       min_df -> 50
#       max_df -> 6000
#       best   -> 0.71
# thats results to be the same respect to f1-weighted-score used in the first approach to model tunning.
#
#Observation:
#in sktlearn->classification_report f1-micro-score is + to accuracy.