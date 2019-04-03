
# coding: utf-8

# In[ ]:
import kevin_utils as util

# In[ ]:


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
    print 'Total words is', str(total_num_words)
    print 'Calculated', len(background_probabilities), 'background probabilities'
    return background_probabilities

# In[ ]:


def word_probabilities(doc):
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
    doc_probabilities = word_probabilities(doc2['aggregated'])
    average_probability = 0
    doc1_split = doc1['aggregated'].split()
    for word in doc1_split:
        average_probability = average_probability + (prior_weight * prior[word])
        if word in doc_probabilities:
            average_probability = average_probability + (doc_probabilities[word] * (1 - prior_weight))
        #print 'Average probability is now', average_probability, prior[word]
    if len(doc1_split) <= 0:
        average_probability = 0
    else:
        average_probability /= len(doc1_split)
    #print len(doc1_split), average_probability
    if 'lecture_transcript' in doc2:
        return average_probability * (len(doc2['lecture_transcript'].split())/longest_transcript)
    else:
        return average_probability * 0.5



# In[1]:


def main(prior_weight):
    corpus = util.get_json('courses_json_preprocessed.json')
    corpus = create_corpus(corpus)
    longest_transcript = get_longest_transcript(corpus)
    similarities = []
    prior = train_background_model(corpus)
    ordering = []
    for course in corpus:
        for lecture in corpus[course]:
            for slide in corpus[course][lecture]:
                ordering.append(course + '!!' + lecture + '!!' + slide)
    num_zeros = 0
    for slide in ordering:
        dict_entries = slide.split('!!')
        for slide2 in ordering:
            dict_entries2 = slide2.split('!!')
            probabilistic_model_output = bayes(corpus[dict_entries[0]][dict_entries[1]][dict_entries[2]], 
                corpus[dict_entries2[0]][dict_entries2[1]][dict_entries2[2]], longest_transcript, prior, prior_weight)
            if probabilistic_model_output == 0:
                num_zeros+=1
            similarities.append(probabilistic_model_output)
        print 'Finished', slide
    print str(num_zeros)
    similarities = util.np.array(similarities).reshape(len(ordering), len(ordering))
    util.write_ordering('probabilistic_model_order', ordering)
    util.write_to_matrix('probabilistic_model_matrix', similarities)



# In[ ]:


weight = 0.2
main(weight)

