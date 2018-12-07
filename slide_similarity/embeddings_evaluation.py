
# coding: utf-8

# In[ ]:


import numpy as np
import json
import nltk
import scipy.spatial.distance as distance


# In[ ]:


def get_json(filename):
    with open(filename, 'r', encoding = 'utf-8') as f:
        return json.load(f)


# In[ ]:


def get_distances(embeddings):
    similarities = {}
    for course in embeddings:
        similarities[course] = {}
        for lecture in embeddings[course]:
            similarities[course][lecture] = {}
            for slide in embeddings[course][lecture]:
                similarities[course][lecture][slide] = {}
                for content in embeddings[course][lecture][slide]:
                    similarities[course][lecture][slide][content] = {}
                    for course2 in embeddings:
                        for lecture2 in embeddings[course2]:
                            for slide2 in embeddings[course2][lecture2]:
                                if content in embeddings[course2][lecture2][slide2]:
                                    similarities[course][lecture][slide][content][course2 + '##' + lecture2 + '##' + slide2] = distance.cosine(
                                        embeddings[course][lecture][slide][content], embeddings[course2][lecture2][slide2][content])
                                    if len(similarities[course][lecture][slide][content]) > 10:
                                        del similarities[course][lecture][slide][content][min(d, key=d.get)]
    return similarities


# In[ ]:


def write_to_json(filename, data):
    with open(filename + '.json', 'w', encoding = 'utf-8') as outfile:
        json.dump(data, outfile)


# In[ ]:


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


# In[ ]:


def main():
    embeddings = get_json('multiplicative_embeddings.json')
    similarities = get_distances(embeddings)
    write_to_json('multiplicative_embeddings_analysis.json')
    make_json_readable('multiplicative_embeddings_analysis.json', 'multiplicative_embeddings_analysis_readable.txt')


# In[ ]:


main()

