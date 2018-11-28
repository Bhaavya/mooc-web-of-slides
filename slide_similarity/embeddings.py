
# coding: utf-8

# In[12]:


import gensim
from gensim.models import KeyedVectors
import numpy as np
import json


# In[13]:


def loadGloveModel(gloveFile):
    model = {}
    with open(gloveFile,'r', encoding = 'utf-8') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print('got embeddings for', len(model), 'words')
    return model


# In[14]:


glove_model = loadGloveModel('glove.6B.300d.txt')


# In[15]:


def get_json(filename):
    with open(filename, 'r', encoding = 'utf-8') as f:
        return json.load(f)


# In[16]:


def get_glove_embeddings(word):
    if word in glove_model:
        return glove_model[word]
    else:
        return np.zeros(300)


# In[17]:


def get_embeddings(json_file, embedding_type):
    embeddings = {}
    for course in json_file:
        if course not in embeddings:
            embeddings[course] = {}
        for lecture in json_file[course]:
            if lecture not in embeddings[course]:
                embeddings[course][lecture] = {}
            for slide in json_file[course][lecture]:
                if slide not in embeddings[course][lecture]:
                    embeddings[course][lecture][slide] = {}
                for content in json_file[course][lecture][slide]:
                    if content not in embeddings[course][lecture][slide]:
                        embeddings[course][lecture][slide][content] = np.zeros(300)
                    splitContent = json_file[course][lecture][slide][content].split()
                    phrase_embeddings = []
                    for x in range(len(splitContent)):
                        glove_embeddings = get_glove_embeddings(splitContent[x])
                        if x==0:
                            phrase_embeddings = np.array(glove_embeddings)
                        elif embedding_type == 2 and glove_embeddings[0] != 0:
                            phrase_embeddings *= np.array(glove_embeddings)
                        else:
                            phrase_embeddings += np.array(glove_embeddings)
                    if embedding_type == 1 and len(splitContent) > 0:
                        for x in range(len(phrase_embeddings)):
                            phrase_embeddings[x] /= len(splitContent)
                    embeddings[course][lecture][slide][content] = []
                    for x in range(len(phrase_embeddings)):
                        embeddings[course][lecture][slide][content].append(phrase_embeddings[x])
            print('finished lecture',lecture)
    return embeddings


# In[18]:


def write_to_json(filename, data):
    with open(filename + '.json', 'w', encoding = 'utf-8') as outfile:
        json.dump(data, outfile)


# In[19]:


def make_json_readable(jsonname, filename):
    data = {}
    with open(jsonname, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    with open(filename, 'w', encoding = 'utf-8') as f:
        f.write('')
    with open(filename, 'a', encoding = 'utf-8') as f:
        for course in data:
            f.write(course + '{\n')
            for lecture in data[course]:
                f.write('\t' + lecture + '{\n')
                for slide in data[course][lecture]:
                    f.write('\t\t' + slide + '{\n')
                    for content in data[course][lecture][slide]:
                        f.write('\t\t\t' + content + ':' + str(np.array(data[course][lecture][slide][content])) + '\n')
                    f.write('\t\t}\n')
                f.write('\t}\n')
            f.write('}\n')


# In[20]:


def main():
    json_file = get_json('courses_json_preprocessed.json')
    additive_embeddings = get_embeddings(json_file, 0)
    write_to_json('additive_embeddings', additive_embeddings)
    average_embeddings = get_embeddings(json_file, 1)
    write_to_json('average_embeddings', average_embeddings)
    multiplicative_embeddings = get_embeddings(json_file, 2)
    write_to_json('multiplicative_embeddings', multiplicative_embeddings)
    make_json_readable('additive_embeddings.json', 'additive_embeddings_readable.txt')
    make_json_readable('average_embeddings.json', 'average_embeddings_readable.txt')
    make_json_readable('multiplicative_embeddings.json', 'multiplicative_embeddings_readable.txt')


# In[21]:


main()

