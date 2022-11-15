import pandas as pd
from featureExctraction_Cleaning_Encoding import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import datetime
import shutil

# print(tf.config.list_physical_devices('GPU'))
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
print('Deleting last logs... \n')
p_dir = './logs/fit'
shutil.rmtree(p_dir)

print('Loading data... \n')
pd.set_option("display.max_columns", None)
train = pd.DataFrame(pd.read_json('train.json/train.json'))
#test = pd.DataFrame(pd.read_json('test.json/test.json'))

print('Cleaning data... \n')
train = pd.DataFrame(ingredientsExtraction(train, 'extraction'))
#test  = data = pd.DataFrame(ingredientsExtraction(test, 'extraction'))

X = pd.Series(train['ingredients'])
t = pd.Series(train['cuisine'])
#X_test = pd.Series(test['ingredients'])
#t_test = pd.Series(test['cuisine'])

target_names = t.drop_duplicates().array

print('Encoding labels...\n')
t, le = encodeCusine(t)
t = pd.Series(t)

print('Splitting dataset (train, validation, test)... \n')
X_train, X_temp, t_train, t_temp = train_test_split( X, t, test_size=0.2, random_state=42)

print('Feature extraction...\n')
vectorizer = TfidfVectorizer(min_df=50, max_df=6000, ngram_range=(1, 2))
tfidf_wm = vectorizer.fit_transform(X_train)

tfidf_tokens = vectorizer.get_feature_names()
X_train = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)

tfidf_test  = vectorizer.transform(X_temp)
X_temp = pd.DataFrame(data = tfidf_test.toarray(),columns = tfidf_tokens)

print('Dataframe to Numpy array...\n')
X_train = X_train.to_numpy()
X_temp = X_temp.to_numpy()

t_train = np.array(t_train)
t_temp  = np.array(t_temp)

print('labels to one hot encoded...\n')
t_train = tf.keras.utils.to_categorical(t_train)
t_temp  = tf.keras.utils.to_categorical(t_temp)
print(t_train)
print(X_train.shape)

print('Trying tensor flow 2...\n')
#settings variable
NB_LABELS = 20
RESHAPE = X_train.shape[1]
NB_HIDDEN = 128
BATCH_SIZE = 128
EPOCHS = 500 #SGD converg close to 400  epochs
VERBOSE = 1
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3
model = tf.keras.models.Sequential()
#input layer
model.add(keras.layers.Dense(NB_HIDDEN,
                             input_shape=(RESHAPE,),
                             name='dense_layer',
                             activation='relu'))
#drop_out_1
model.add(keras.layers.Dropout(DROPOUT))
#hidden layer 1
model.add(keras.layers.Dense(NB_HIDDEN,
                             name= 'dense_layer_2',
                             activation='relu'))
#drop_out_2
model.add(keras.layers.Dropout(DROPOUT))
#output layer
model.add(keras.layers.Dense(NB_LABELS,
                             name= 'dense_layer_3',
                             activation='softmax'))
#model summary
model.summary()

#compiling model
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics= [tf.keras.metrics.BinaryAccuracy(),
                      tfa.metrics.F1Score(num_classes=NB_LABELS,
                      average='micro',
                      threshold=0.5)])
#tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#training
model.fit(X_train,t_train,
          batch_size= BATCH_SIZE,
          epochs=EPOCHS,
          verbose=VERBOSE,
          validation_split=VALIDATION_SPLIT,
          callbacks=[tensorboard_callback])
#tensorboard check
#use this command on terminal: tensorboard --logdir logs/fit

#evaluation
test_loss, test_acc, test_f1_micro = model.evaluate(X_temp,t_temp,callbacks=tensorboard_callback)
print('Test Accuracy:', test_f1_micro)