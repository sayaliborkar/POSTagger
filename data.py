import nltk
import numpy as np
import pickle

nltk.download('treebank')
nltk.download('universal_tagset')
d = nltk.corpus.treebank.tagged_sents(tagset = 'universal')
d = np.asarray(d)
print d.shape

words = []
tags = []

for i in range(d.shape[0]):
	w = []
	t = []
	for j in range(len(d[i])):
		w.append(str(d[i][j][0]))
		t.append(str(d[i][j][1]))
	
	words.append(w)
	tags.append(t)

print words
print tags

pickle.dump(words,open('words.p','wb'))
pickle.dump(tags,open('tags.p','wb'))

unique_words = set()
for line in words:
	for w in line:
		unique_words.add(w)
print len(unique_words)

unique_tags = set()
for line in tags:
	for t in line:
		unique_tags.add(t)
print len(unique_tags)

word_to_id = {word: i for i, word in enumerate(unique_words, start=1)}
pos_to_id = {pos: i for i, pos in enumerate(unique_tags, start=1)}

id_to_word = {v: k for k, v in word_to_id.items()}
id_to_pos = {v: k for k, v in pos_to_id.items()}

dicts = [word_to_id,pos_to_id,id_to_word,id_to_pos]
with open('vocab.p', 'wb') as f:
	pickle.dump(dicts, f)

X = []
for line in words:
	x = []
	for w in line:
		x.append(word_to_id[w])
	X.append(x)
print X

Y = []
for line in tags:
	y = []
	for t in line:
		y.append(pos_to_id[t])
	Y.append(y)
print Y

pickle.dump(X,open('X.p','wb'))
pickle.dump(Y,open('Y.p','wb'))