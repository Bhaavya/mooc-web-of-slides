
# coding: utf-8

# In[167]:

import kevin_utils as util

vectorizer = util.TfidfVectorizer()
glove_model = util.loadGloveModel('glove.6B.300d.txt')
word2vec_model = util.loadWord2Vec('slides_wordvectors.kv')


def create_corpus(corpus_json):
    corpus = []
    ordering = []
    for course in corpus_json:
        for lecture in corpus_json[course]:
            for slide in corpus_json[course][lecture]:
                aggregated = ''
                for content in corpus_json[course][lecture][slide]:
                    aggregated += corpus_json[course][lecture][slide][content]
                corpus.append(aggregated)
                ordering.append(course + '!!' + lecture + '!!' + slide)
    print 'Finished aggregating'
    return (corpus, ordering)


def get_embeddings(corpus, tfidf_matrix, ordering, word_indices, is_glove):
    embeddings = {}
    for y in range(len(corpus)):
        splitContent = corpus[y].split()
        phrase_embeddings = util.np.zeros(300)
        for x in range(len(splitContent)):
            embedding = util.np.zeros(300)
            if is_glove:
                embedding = util.get_glove_embeddings(splitContent[x], glove_model)
            else:
                embedding = util.get_word2vec_embeddings(splitContent[x], word2vec_model)
            embedding *= tfidf_matrix[y][word_indices[splitContent[x]]]
            phrase_embeddings += embedding
        if len(splitContent) > 0:
            for x in range(len(phrase_embeddings)):
                phrase_embeddings[x] /= len(splitContent)
        embeddings[ordering[y]] = phrase_embeddings.tolist()
    print "Finished getting embeddings"
    return embeddings


def main(is_glove):
    corpus_json = util.get_json('courses_json_preprocessed.json')
    corpus, ordering = create_corpus(corpus_json)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    word_indices = vectorizer.vocabulary_
    #word_indices = {}
    #for x in range(len(word_ordering)):
    #    word_indices[word_ordering[x]] = x
    embeddings = get_embeddings(corpus, tfidf_matrix.todense().tolist(), ordering, word_indices, is_glove)
    similarities = util.get_distances(embeddings, ordering)
    if is_glove:
        util.write_ordering('tfidf_glove_embeddings_order', ordering)
        util.write_to_matrix('tfidf_glove_embeddings', similarities)
        print "finished with glove"
    else:
        util.write_ordering('tfidf_word2vec_embeddings_order', ordering)
        util.write_to_matrix('tfidf_word2vec_embeddings', similarities)
        


# In[ ]:

main(1)
main(0)

