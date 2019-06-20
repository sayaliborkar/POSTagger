from __future__ import division
import nltk
import pickle
import os
import numpy as np
import pandas as pd
import tensorflow as tf

vocab = pickle.load(open('vocab.p','rb'))
word_to_id = vocab[0]

glove = np.random.random([20000,50])
print glove

with open('glove.6B.50d.txt','r') as file:
	for index, line in enumerate(file):
		values = line.split() # Word and weights separated by space
		word = values[0] # Word is first symbol on each line
		word_weights = np.asarray(values[1:], dtype=np.float32) # Remainder of line is weights for word

		if word in word_to_id.keys():
			for i in range(50):
				glove[word_to_id[word]][i] = word_weights[i]

print glove.shape
print glove[0]
print glove[1]
print glove[2]
print glove[3]
print glove[4]

with open('glove.p', 'wb') as f:
	pickle.dump(glove, f)

f.close()