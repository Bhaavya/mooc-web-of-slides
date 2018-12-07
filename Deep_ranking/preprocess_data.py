import numpy as np
import os
import nltk
import itertools
import io
import csv
import itertools
from sklearn import preprocessing
## create directory to store preprocessed data
if(not os.path.isdir('preprocessed_data')):
    os.mkdir('preprocessed_data')

## get all of the training reviews (including unlabeled reviews)
train_test_file = '../slide_similarity/tmp/input2.txt'
train_test_names =  '../slide_similarity/tmp/slide_names2.txt'
count = 0
X = []

with io.open(train_test_file,'r',encoding = 'utf-8') as f:
    lines = f.readlines()
with io.open(train_test_names,'r',encoding = 'utf-8') as f:
    names = f.readlines()
i=0
for line in lines:
    #print type(line)
    #line = line.replace('\x96',' ')   #should remove lot of characters here.
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]
    line =filter(lambda l: len(l)>1, line)
    X.append((names[i].strip(),line))
    i = i+1



## number of tokens per slide stats to select sequence length for LSTM
no_of_tokens = []
for tokens in X:
    no_of_tokens.append(len(tokens[1]))
no_of_tokens = np.asarray(no_of_tokens)
print('Total: ', np.sum(no_of_tokens), ' Min: ', np.min(no_of_tokens), ' Max: ', np.max(no_of_tokens), ' Mean: ', np.mean(no_of_tokens), ' Std: ', np.std(no_of_tokens))


just_tokens = map(lambda l : l[1], X)
### word_to_id and id_to_word. associate an id to every unique token in the training data
all_tokens = itertools.chain.from_iterable(just_tokens)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

all_tokens = itertools.chain.from_iterable(just_tokens)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)


## let's sort the indices by word frequency instead of random
x_train_token_ids = [[word_to_id[token] for token in x[1]] for x in X]
count = np.zeros(id_to_word.shape)
for x in x_train_token_ids:
    for token in x:
        count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]
#histogram of words
hist = np.histogram(count,bins=[1,10,100,1000,10000])
print(hist)
for i in range(10):
    print(id_to_word[i],count[i])

## recreate word_to_id based on sorted list
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}
## assign -1 if token doesn't appear in our dictionary
## add +1 to all token ids, we went to reserve id=0 for an unknown token
X_names_tokens = [(x[0], [word_to_id.get(token,-1)+1 for token in x[1]]) for x in X]


## save dictionary
np.save('preprocessed_data/slide_dictionary.npy',np.asarray(id_to_word))

## save training data to single text file
with io.open('preprocessed_data/X.txt','w',encoding='utf-8') as f:
    for tokens in X_names_tokens:
        for token in tokens[1]:
            f.write(str(token).encode("utf-8").decode("utf-8")+' ')
        f.write(("\n").encode("utf-8").decode("utf-8")) 

with io.open('preprocessed_data/X_names.txt','w',encoding='utf-8') as f:
    for tokens in X_names_tokens:
        f.write(tokens[0])
        f.write(("\n").encode("utf-8").decode("utf-8"))

#preprocess similarity matrices
def preprocess_matrices():
    string_list = list(csv.reader(open("../data/average_embeddings_aggregated_matrix.csv", "rb"), delimiter=","))
    embedding_matrix = []
    for i in range(len(string_list)):
        #print string_list[i]
        embedding_matrix += [np.array(string_list[i][:-1]).astype("float")]
    embedding_matrix = np.nan_to_num(np.array(embedding_matrix))
    embeddings_order =  list(csv.reader(open("../data/average_embeddings_aggregated_order.csv", "rb"), delimiter=","))
    (x,y) = embedding_matrix.shape
    embeddings_order = [s[0].replace('!!','##') for s in embeddings_order]
    embeddings_order_for_bert = ['##'.join(s.split('##')[1:]) for s in embeddings_order]
    embeddings_matrix = preprocessing.normalize(np.reshape(embedding_matrix, (-1,1)), norm='l2')
    embedding_matrix = np.reshape(embeddings_matrix, (x,y))
    bert_sim = np.load('../data/bert/bert_sim.npy')
    bert_order = open('../data/bert/slides_names_bert.txt','r').readlines()
    bert_order = [b.strip() for b in bert_order]
    bert_sim = preprocessing.normalize(np.reshape(bert_sim, (-1,1)), norm='l2')
    bert_sim = np.reshape(bert_sim, (x,y))
    '''
    x1= bert_order.index(embeddings_order_for_bert[1])
    y1=bert_order.index(embeddings_order_for_bert[0])
    print (bert_sim[x1][y1])
    print bert_sim.shape
    '''
    bert_embeddings_indices = []
    for x in embeddings_order_for_bert:
        bert_embeddings_indices += [bert_order.index(x)]
    bert_sim = bert_sim[bert_embeddings_indices,:]
    bert_sim = bert_sim[:,bert_embeddings_indices]    
    print (bert_sim[1][0])

    tfidf_sim = np.load('../data/tfidf-similarity.npy')
    tfidf_order = open('../data/slides_names2.txt','r').readlines()
    tfidf_sim = preprocessing.normalize(np.reshape(tfidf_sim, (-1,1)), norm='l2')
    tfidf_sim = np.reshape(tfidf_sim, (x,y))
    '''
    x1= tfidf_order.index(embeddings_order[1])
    y1=tfidf_order.index(embeddings_order[0])
    print (bert_sim[x1][y1])
    print bert_sim.shape
    '''
    tfidf_embeddings_indices = []
    for x in embeddings_order_for_bert:
        tfidf_embeddings_indices += [bert_order.index(x)]
    tfidf_sim = tfidf_sim[tfidf_embeddings_indices,:]
    tfidf_sim = tfidf_sim[:,tfidf_embeddings_indices]
    return (bert_sim, embedding_matrix, tfidf_sim, embeddings_order)


def combine_matrices(alphas, matrices):
    #reorder_embeddings:
    final_matrix = sum([alphas[i]*matrices[i] for i in range(len(alphas))])
    final_matrix = np.round(final_matrix,6)
    final_matrix = np.asarray(final_matrix,dtype = float)
    final_matrix[final_matrix==0] =  0.00001
    prob_sum = np.sum(final_matrix[np.tril_indices(final_matrix.shape[0])])
    final2_matrix = float(1)/final_matrix
    final2_matrix = np.round(final2_matrix,6)
    prob2_sum = np.sum(final2_matrix[np.tril_indices(final2_matrix.shape[0])])
    l = final2_matrix.shape[0]
    final_matrix = preprocessing.normalize(final_matrix,norm = 'l1')
    final2_matrix = preprocessing.normalize(final2_matrix,norm = 'l1')

    '''
    for i in range(l):
        final_matrix[i,:] = final_matrix[i,:]/np.sum(final_matrix[i,:])
        final_matrix[i,:] = np.round(final_matrix[i,:],8)
        x = 1.0-np.sum(final_matrix[i,:])
        final_matrix[i,0] = final_matrix[i,0] + x
        print np.sum(final_matrix[i,:]) 
    for i in range(l):
        final2_matrix[i,:] = final2_matrix[i,:]/np.sum(final2_matrix[i,:])
        final2_matrix[i,:] = np.round(final2_matrix[i,:],6)
        x = 1.0-np.sum(final2_matrix[i,:])
        final2_matrix[i,0] = final2_matrix[i,0] + x
        print np.sum(final2_matrix[i,:])
    '''
    #print prob_sum
    #print prob2_sum
    return (final_matrix, final2_matrix)
'''
glove_filename = '../data/glove.840B.300d.txt'
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
# added a vector of zeros for the unknown tokens
glove_embeddings = np.concatenate((np.zeros((1,300)),glove_embeddings))

word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}

X_names_tokens = [(x[0],[word_to_id.get(token,-1)+1 for token in x[1]]) for x in X_names_tokens]

np.save('preprocessed_data/glove_dictionary.npy',glove_dictionary)
np.save('preprocessed_data/glove_embeddings.npy',glove_embeddings)

with io.open('preprocessed_data/imdb_train_glove.txt','w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

with io.open('preprocessed_data/imdb_test_glove.txt','w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
'''