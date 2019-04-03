
# coding: utf-8

# In[ ]:
import kevin_utils as util
from sklearn.preprocessing import MinMaxScaler
# In[ ]:
scaler = MinMaxScaler()

def load_numpy(filename):
    return util.np.load(filename)

def load_csv(filename, ordering):
    similarities = []
    x = 0
    with open(filename, 'r') as f:
        for line in f:
            similarities.append([])
            for value in line.split('\n')[0].split(','):
                if value != '':
                    if value == 'nan':
                        similarities[x].append(0)
                    else:
                        similarities[x].append(float(value))
            x += 1
    print 'Finished loading', filename
    return similarities

def load_order(filename):
    ordering = []
    with open(filename, 'r') as f:
        for line in f:
            ordering.append(line.split('\n')[0])
    return ordering
# In[ ]:


def get_largest_difference(corpus):
    largest_difference = 0
    for slide in corpus:
        for slide2 in corpus:
            difference = abs(len(corpus[slide].split()) - len(corpus[slide2].split()))
            if difference > largest_difference:
                largest_difference = difference
    print 'largest difference is', str(largest_difference)
    return largest_difference

# In[ ]:


def create_corpus(corpus_json, title2num):
    x = 0
    new_corpus = {}
    for course in corpus_json:
        for lecture in corpus_json[course]:
            for slide in corpus_json[course][lecture]:
                aggregated = ''
                for content in corpus_json[course][lecture][slide]:
                    aggregated += corpus_json[course][lecture][slide][content]
                new_corpus[title2num[course + '##' + lecture + '##' + slide]] = aggregated
                if aggregated != '':
                    x += 1
    print 'Finished aggregating', x, 'slides'
    return new_corpus


def matrix2dict(ordering, similarity, title2num):
    similarity = scaler.fit_transform(similarity)
    result_dict = {}
    for x in range(len(ordering)):
        result_dict[title2num[ordering[x]]] = {}
        for y in range(len(ordering)):
            result_dict[title2num[ordering[x]]][title2num[ordering[y]]] = similarity[x][y]
    return result_dict

def ensemble(embedding_dict, tfidf_dict, title_dict, corpus, largest_difference):
    new_similarity = {}
    for slide in corpus:
        new_similarity[slide] = {}
        for slide2 in corpus:
            difference = abs(len(corpus[slide].split()) - len(corpus[slide2].split()))
            tfidf_weighting = difference/largest_difference
            new_similarity[slide][slide2] = (tfidf_weighting * tfidf_dict[slide][slide2] + 0.5 * title_dict[slide][slide2] + 
                0.5 * embedding_dict[slide][slide2]) / (tfidf_weighting + 1)
        print 'Finished ensemble modeling', slide
    return new_similarity


def main():
    title_ordering = load_order('./data/X_names.txt')
    title2num = {}
    for x in range(len(title_ordering)):
        title2num[title_ordering[x]] = x

    corpus = util.get_json('courses_json_preprocessed.json')
    corpus = create_corpus(corpus, title2num)
    largest_difference = get_largest_difference(corpus)

    embeddings_ordering = load_order('./data/word2vec_embeddings_order.txt')
    embeddings_similarities = load_csv('./data/word2vec_embeddings.csv', embeddings_ordering)
    embeddings_similarities = matrix2dict(embeddings_ordering, embeddings_similarities, title2num)

    tfidf_ordering = load_order('./data/slides_names_tfidf.txt')
    tfidf_similarities = load_numpy('./data/tfidf_sim.npy')
    tfidf_similarities = matrix2dict(tfidf_ordering, tfidf_similarities, title2num)

    
    title_similarities = load_numpy('./data/title_similarity.npy')
    title_similarities = matrix2dict(title_ordering, title_similarities, title2num)

    similarities = ensemble(embeddings_similarities, tfidf_similarities, title_similarities, corpus, largest_difference)
    util.write_ordering('ensemble_order', title_ordering)
    similarity_matrix = []
    for x in range(len(title_ordering)):
        for y in range(len(title_ordering)):
            similarity_matrix.append(similarities[x][y])
    similarity_matrix = util.np.array(similarity_matrix).reshape(len(title_ordering), len(title_ordering))
    util.np.save('ensemble_matrix', similarity_matrix)



# In[ ]:


main()

