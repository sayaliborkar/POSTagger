from __future__ import division
import nltk
import pickle
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.utils import shuffle

from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential

X = pickle.load(open('X.p','rb'))
y = pickle.load(open('Y.p','rb'))

X = pad_sequences(X)
y = pad_sequences(y)

MAX_SEQUENCE_LENGTH = X.shape[1]

X_train = X[0:3000]
y_train = y[0:3000]

X_test = X[3000:]
y_test = y[3000:]

vocab = pickle.load(open('vocab.p','rb'))
word_to_id = vocab[0]
pos_to_id = vocab[1]
id_to_word = vocab[2]
id_to_pos = vocab[3]

n_tags = len(id_to_pos)

n_train_samples = X_train.shape[0]

embedding_matrix = pickle.load(open('glove.p','rb'))

embedded_train = []
for line in X_train:
	sen = []
	for i in line:
		sen.append(embedding_matrix[i])
	embedded_train.append(sen)
embedded_train = np.asarray(embedded_train)
print embedded_train.shape

y_tr = to_categorical(y_train, num_classes=n_tags+1)
print y_tr.shape

model = Sequential()
# model.add(LSTM(50, input_shape=(272,50), return_sequences=True))
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(MAX_SEQUENCE_LENGTH, 50)))
model.add(TimeDistributed(Dense(y_tr.shape[2], activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

# TRAIN
model.fit(embedded_train, y_tr, epochs=20, batch_size=30, verbose=1)

name = 'model.h5'
model.save(name)
print('MODEL SAVED!!')

embedded_test = []
for line in X_test:
	sen = []
	for i in line:
		sen.append(embedding_matrix[i])
	embedded_test.append(sen)
embedded_test = np.asarray(embedded_test)
print embedded_test.shape

y_te = to_categorical(y_test, num_classes=n_tags+1)
print y_te.shape

test_results = model.evaluate(embedded_test, y_te, verbose=0)
print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))

# TEST
model1 = load_model('model.h5')

prediction = model1.predict(embedded_test)
print prediction.shape

print prediction[0][260]
print y_test[0][260]


conf = np.zeros((n_tags+1,n_tags+1))
print conf

count = 0
total = 0
for i in range(X_test.shape[0]):
	label = []
	preds = []
	for word in y_test[i]:
		if word == 0:
			continue
		label.append(id_to_pos[word])
	for pred in prediction[i]:
		try:
			preds.append(id_to_pos[np.argmax(pred)])
		except:
			pass
	if not(len(label) == len(preds)):
		continue
	for j in range(len(label)):
		print str(label[j]) + " " + str(preds[j])
		conf[pos_to_id[label[j]]][pos_to_id[preds[j]]]+= 1
		total+=1
		if label[j] == preds[j]:
			count+=1
print count
print total

conf1 = open('conf-1.p','wb')
pickle.dump(conf,conf1)