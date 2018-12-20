from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from scipy.stats import entropy
import numpy as np

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

actual_slides = []
with open('./data/actual_slides.txt', 'r') as f:
	for line in f:
		actual_slides.append(line)

choose_slides = []

with open('./data/slide_names3.txt', 'r') as f:
	for line in f:
		if line in actual_slides:
			choose_slides.append(1)
		else:
			choose_slides.append(0)

documents = []
with open('./data/input3.txt', 'r') as f:
	x = 0
	for line in f:
		if choose_slides[x]:
			documents.append(line)
		x += 1

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 0
with open('./data/slide_names3.txt', 'r') as f:
	last_week = ''
	for line in f:
		splitLine = line.split('##')
		if splitLine[2] != last_week:
			no_topics += 1
			last_week = splitLine[2]

print 'There are', no_topics, 'topics'


# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit_transform(tfidf)

# Run LDA
#lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit_transform(tf)

#no_top_words = 10
#display_topics(nmf, tfidf_feature_names, no_top_words)
#display_topics(lda, tf_feature_names, no_top_words)

def kl_divergence(topic_array):
	kl_divergences = []
	for x in range(len(topic_array)):
		for slide2 in topic_array:
			kl_divergences.append(entropy(topic_array[x], slide2))
		print 'Finished KL divergence for slide', x
	kl_divergences = np.array(kl_divergences).reshape(len(topic_array), len(topic_array))
	return kl_divergences

def write_csv(filename, array):
	with open(filename + '.csv', 'a') as f:
		for x in range(len(array)):
			for y in range(len(array)):
				f.write(str(array[x][y]) + ',')
			f.write('\n')

#lda_kl_divergence = kl_divergence(lda)

#np.save('LDA_KL_Divergence', lda_kl_divergence)
#write_csv('LDA_KL_Divergence', lda_kl_divergence)

nmf_kl_divergence = kl_divergence(nmf)

np.save('NMF_KL_Divergence', nmf_kl_divergence)
write_csv('NMF_KL_Divergence', nmf_kl_divergence)