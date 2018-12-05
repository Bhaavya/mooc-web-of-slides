
# coding: utf-8

# In[ ]:


import json
import numpy as np


# In[ ]:


def get_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


# In[ ]:


def get_longest_transcript(corpus_json):
    longest_transcript = 0
    for course in corpus_json:
        for lecture in corpus_json[course]:
            for slide in corpus_json[course][lecture]:
                if 'lecture_transcript' in corpus_json[course][lecture][slide]:
                    if len(corpus_json[course][lecture][slide]['lecture_transcript']) > longest_transcript:
                        longest_transcript = len(corpus_json[course][lecture][slide]['lecture_transcript'])
    print 'Longest transcript is', str(longest_transcript)
    return longest_transcript


def train_background_model(corpus_json):
    background_probabilities = {}
    total_num_words = 0
    for course in corpus_json:
        for lecture in corpus_json[course]:
            for slide in corpus_json[course][lecture]:
                for word in corpus_json[course][lecture][slide]['aggregated'].split():
                    if word not in background_probabilities:
                        background_probabilities[word] = 1
                    else:
                        background_probabilities[word] += 1
                    total_num_words += 1
    for word in background_probabilities:
        background_probabilities[word] /= total_num_words
    print 'Calculated', len(background_probabilities), 'background probabilities'
    return background_probabilities

# In[ ]:


def word_probabilities(doc, prior, prior_weight):
    words = doc.split()
    probabilities = {}
    for word in words:
        if word not in probabilities:
            probabilities[word] = 1
        else:
            probabilities[word] += 1
    doc_length = len(words)
    for probability in probabilities:
        probabilities[probability] /= doc_length
        probabilities[probability] *= (1 - prior_weight)
        probabilities[probability] += (prior_weight * prior[probability])
    return probabilities


# In[ ]:


def create_corpus(corpus_json):
    x = 0
    for course in corpus_json:
        for lecture in corpus_json[course]:
            for slide in corpus_json[course][lecture]:
                aggregated = ''
                for content in corpus_json[course][lecture][slide]:
                    aggregated += corpus_json[course][lecture][slide][content]
                corpus_json[course][lecture][slide]['aggregated'] = aggregated
                if aggregated != '':
                    x += 1
    print 'Finished aggregating', x, 'slides'
    return corpus_json


# In[ ]:


def bayes(doc1, doc2, longest_transcript, prior, prior_weight):
    doc_probabilities = word_probabilities(doc2['aggregated'], prior, prior_weight)
    average_probability = 0
    doc1_split = doc1['aggregated'].split()
    for word in doc1_split:
        if word in doc_probabilities:
            average_probability += doc_probabilities[word]
    if len(doc1_split) <= 0:
        average_probability = 0
    else:
        average_probability /= len(doc1_split)
    if 'lecture_transcript' in doc1:
        return average_probability * (len(doc1['lecture_transcript'].split())/longest_transcript)
    else:
        return average_probability * 0.5

# In[ ]:


def write_to_json(filename, data):
    with open(filename + '.json', 'w') as outfile:
        json.dump(data, outfile)


# In[ ]:


def make_json_readable(jsonname, filename):
    data = {}
    with open(jsonname, 'r') as f:
        data = json.load(f)
    with open(filename, 'w') as f:
        f.write('')
    with open(filename, 'a') as f:
        for course in data:
            f.write(course + '{\n')
            for lecture in data[course]:
                f.write('\t' + lecture + '{\n')
                for slide in data[course][lecture]:
                    f.write('\t\t' + slide + '{\n')
                    for key, value in sorted(data[course][lecture][slide].iteritems(), key=lambda (k,v): (v,k)):
                         f.write('\t\t\t' + str(key) + ':' + str(value) + '\n')
                    f.write('\t\t}\n')
                f.write('\t}\n')
            f.write('}\n')


# In[1]:


def main(prior_weight):
    corpus = get_json('courses_json_preprocessed.json')
    corpus = create_corpus(corpus)
    longest_transcript = get_longest_transcript(corpus)
    similarities = {}
    prior = train_background_model(corpus)
    for course in corpus:
        similarities[course] = {}
        for lecture in corpus[course]:
            similarities[course][lecture] = {}
            for slide in corpus[course][lecture]:
                similarities[course][lecture][slide] = {}
                for course2 in corpus:
                    for lecture2 in corpus[course2]:
                        for slide2 in corpus[course2][lecture2]:
                            probabilistic_model_output = bayes(corpus[course][lecture][slide], corpus[course2][lecture2][slide2], longest_transcript, prior, prior_weight)
                            if probabilistic_model_output != 0:
                                similarities[course][lecture][slide][course2 + '!!' + lecture2 + '!!' + slide2] = probabilistic_model_output
                            if len(similarities[course][lecture][slide]) > 10:
                                del similarities[course][lecture][slide][min(similarities[course][lecture][slide], key=similarities[course][lecture][slide].get)]
        print 'Finished', course
    write_to_json('probabilities_model', similarities)
    make_json_readable('probabilities_model.json', 'probabilities_model_readable.txt')


# In[ ]:


weight = 0.2
main(weight)

