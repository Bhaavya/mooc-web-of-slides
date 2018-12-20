import numpy as np
import json
import nltk
import io
import scipy.spatial.distance as distance
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def get_json(filename):
    with open(filename, 'r', encoding = 'utf-8') as f:
        return json.load(f, encoding = 'utf-8')

def loadGloveModel(gloveFile):
    model = {}
    with io.open(gloveFile,'r', encoding = 'utf8') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print('got embeddings for', len(model), 'words')
    return model

def get_glove_embeddings(word, glove_model):
    if word in glove_model:
        return glove_model[word]
    else:
        return np.zeros(300)


def loadWord2Vec(word2vecFile):
    wv = KeyedVectors.load(word2vecFile,mmap = 'r')
    model = {}
    for word in wv.vocab:
        model[word.encode('utf8').decode('utf8')] = np.zeros(len(wv[word]))
        model[word.encode('utf8').decode('utf8')] += wv[word]
    print('got embeddings for', len(model), 'words')
    return model

def get_word2vec_embeddings(word, word2vec):
    if word in word2vec:
        return word2vec[word]
    else:
        return np.zeros(300)

def get_distances(embeddings):
    similarities = []
    for x in range(len(embeddings)):
        for y in range(len(embeddings)):
            if len(embeddings[x]) > 0 and len(embeddings[y]) > 0:
                similarities.append(1 - distance.cosine(embeddings[x], embeddings[y]))
            else:
                similarities.append(-1)
        print('Finished similarity for', x)
    similarities = scaler.fit_transform(np.array(similarities).reshape(len(embeddings), len(embeddings)))
    return similarities


def write_ordering(filename, data):
    with open(filename + '.txt', 'w') as outfile:
        outfile.write('')
    with open(filename + '.txt', 'a') as outfile:
        for slide in data:
            outfile.write(slide + '\n')




def write_to_matrix(filename, data):
    with open(filename + '.csv', 'w') as outfile:
        outfile.write('')
    with open(filename + '.csv', 'a') as outfile:
        for slide in range(len(data)):
            for slide2 in range(len(data[slide])):
                outfile.write(str(data[slide][slide2]) + ',')
            outfile.write('\n')



def write_to_json(filename, data):
    with open(filename + '.json', 'w') as outfile:
        json.dump(data, outfile)

def load_order(filename):
    ordering = []
    with open(filename, 'r') as f:
        for line in f:
            ordering.append(line.split('\n')[0])
    return ordering
