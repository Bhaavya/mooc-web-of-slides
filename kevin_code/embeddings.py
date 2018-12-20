
# coding: utf-8

# In[12]:


import kevin_utils as util

glove_model = util.loadGloveModel('glove.6B.300d.txt')
word2vec_model = util.loadWord2Vec('./data/slides_wordvectors.kv')


def get_embeddings(corpus, is_glove):
    embeddings = []
    for y in range(len(corpus)):
        splitContent = corpus[y].split()
        phrase_embeddings = util.np.zeros(300)
        for x in range(len(splitContent)):
            if is_glove:
                embedding = util.get_glove_embeddings(splitContent[x], glove_model)
            else:
                embedding = util.get_word2vec_embeddings(splitContent[x], word2vec_model)
            phrase_embeddings += embedding
        if len(splitContent) > 0:
            for x in range(len(phrase_embeddings)):
                phrase_embeddings[x] /= len(splitContent)
        embeddings.append(phrase_embeddings.tolist())
    print "Finished getting embeddings"
    return embeddings


def main():
    corpus = util.load_order('./data/input3.txt')
    word2vec_embeddings = get_embeddings(corpus, 0)
    word2vec_similarities = util.get_distances(word2vec_embeddings)
    util.np.save('word2vec_similarities', word2vec_similarities)
    util.write_to_matrix('word2vec_similarities', word2vec_similarities)
    


# In[21]:


main()

