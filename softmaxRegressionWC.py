import pandas as pd
from matplotlib import pyplot as plt

from featureExctraction_Cleaning_Encoding import *
import seaborn as sns; sns.set()
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


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
X_val, X_test, t_val, t_test = train_test_split(X_temp, t_temp, test_size=0.5, random_state= 42)

print('\n Model selection...\n')

C = np.arange(0.2,1.2,0.2)

penalty =["l2", "l1"]

best_score_avg = 0
best_c = 0
best_p = ''

clf = make_pipeline(TfidfVectorizer(min_df=50, max_df=6000, ngram_range=(1, 2))
                    , LogisticRegression( random_state=0, solver='saga', penalty='none', class_weight=None,
                                         multi_class='multinomial'))
clf.fit(X_train, t_train)
labels_val = clf.predict(X_val)
score_avg_none = f1_score(t_val, labels_val, average='weighted')


for p in penalty:
    for c in C:
        clf = make_pipeline(TfidfVectorizer(min_df=50, max_df=6000, ngram_range=(1, 2))
                        , LogisticRegression(C= c ,random_state=0, solver='saga', penalty= p, class_weight=None,
                                             multi_class='multinomial'))
        clf.fit(X_train, t_train)
        labels_val = clf.predict(X_val)
        score_avg = f1_score(t_val, labels_val, average='weighted')
        if score_avg > best_score_avg :
            best_score_avg = score_avg
            best_c = c
            best_p = p

if best_score_avg < score_avg_none:
    best_score_avg = score_avg_none
    best_p = 'none'
    best_c = 1
print(f'best regularization = {best_c}')          #-> 1
print(f'best regularization = {best_p}')          #-> l2
print(f' with f1_weighted of = {best_score_avg}') #-> 0.765

# best_c = 1
# best_p = "l2"

print('\n Testing data on D_test... \n')

clf = make_pipeline(TfidfVectorizer(min_df=50, max_df=6000, ngram_range=(1, 2))
                    , LogisticRegression(C= best_c ,random_state=0, solver='saga', penalty= best_p,
                                         class_weight=None, multi_class='multinomial'))

X_train = pd.concat([X_train, X_val])
t_train = pd.concat([t_train, t_val])

clf.fit(X_train, t_train)
labels = clf.predict(X_test)

print('\n Final Score \n')
print(classification_report(t_test, labels, target_names= target_names))  # -> 0.77


# softmax_score = classification_report(t_test, labels, target_names= target_names, output_dict= True)
# score = pd.DataFrame(softmax_score)
# score = score.T
# score.to_excel("softmax_results.xlsx")


t_test = decodeCusine(t_test, le)
labels = decodeCusine(labels, le)

mat = confusion_matrix(t_test, labels, labels= target_names)
sns.heatmap(mat.T, square = True, annot = True, fmt ='d', cbar = False, xticklabels = target_names, yticklabels = target_names)
plt.title('Confusion Matrix Softmax Regression')
plt.xlabel('true label')
plt.ylabel('predicted labels')
# plt.savefig('softmax_confusionMatrix', bbox_inches='tight')
plt.show()
