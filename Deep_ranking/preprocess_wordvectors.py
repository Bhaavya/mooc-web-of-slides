import io
from gensim.models import KeyedVectors
import numpy as np
'''
glove_filename = '../data/glove.6B.300d.txt'
with io.open(glove_filename,'r',encoding='utf-8') as f:
    lines = f.readlines()

glove_dictionary = []
glove_embeddings = []
count = 0
for line in lines:
    line = line.strip()
    line = line.split(' ')
    glove_dictionary.append(line[0])
    embedding = np.asarray(line[1:],dtype=np.float)
    glove_embeddings.append(embedding)
    count+=1
    if(count>=100000):
        break
glove_dictionary = np.asarray(glove_dictionary)
glove_embeddings = np.asarray(glove_embeddings)

with io.open('../data/glove_word_vectors.txt','w', encoding='utf-8') as f:
    for i in range(len(glove_dictionary)):
		f.write(glove_dictionary[i])
		for j in range(300):
			f.write(' '+str(glove_embeddings[i][j]).encode("utf-8").decode("utf-8")+' ')
		f.write(("\n").encode("utf-8").decode("utf-8"))
'''

def loadWord2Vec(word2vecFile):
   wv = KeyedVectors.load(word2vecFile,mmap = 'r')
   model = {}
   for word in wv.wv.vocab:
       model[word.encode('utf8').decode('utf8')] = np.zeros(len(wv[word]))
       model[word.encode('utf8').decode('utf8')] += wv[word]
   print('got embeddings for', len(model), 'words')
   return model

model = loadWord2Vec("../data/slides_wordvectors_new.kv")

model = model.items()
words = [l[0] for l in model]
embeddings = [l[1] for l in model]
with io.open('../data/word2vec_word_vectors.txt','w', encoding='utf-8') as f:
    for i in range(len(words)):
		f.write(words[i])
		for j in range(300):
			f.write(' '+str(embeddings[i][j]).encode("utf-8").decode("utf-8"))
		f.write(("\n").encode("utf-8").decode("utf-8"))


f = open('../slide_similarity/tmp/input3.txt','r')
slides = f.readlines()
not_present = []
new_slides = []
for slide in slides:
	slide_words = slide.split(' ')
	slide_words = filter(lambda l: l not in words,slide_words)
	not_present += slide_words
	new_slides += [' '.join(slide_words)]
print set(not_present) 
'''
f = open('../LFTM_model/input3.txt','w')
for slide in new_slides:
	f.write(slide)
	f.write('\n')
'''
'''
slide_names = open('../slide_similarity/tmp/slide_names3.txt','r').readlines()
f = open('../LFTM_model/input3.txt','r')
slides  = f.readlines()
i=0
remove_slide = []
for slide in slides:
	slide =slide.strip()
	if (slide == ''):
		remove_slide += [slide_names[i]]
	i = i+1
for r in remove_slide:
	slide_names.remove(r)
f = open('../LFTM_model/slide_names_special3.txt','w')
for slide in slide_names:
	f.write(slide)
'''
