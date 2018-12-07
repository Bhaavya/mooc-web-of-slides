from sklearn.feature_extraction.text import TfidfVectorizer as tv
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf(corpus):
	vec = tv()
	tfidfs = vec.fit_transform(corpus)
	print (tfidfs.shape)
	return (tfidfs,vec)

def compute_tfidf_sim(corpus):
	tfidfs,vec = build_tfidf(corpus)
	return cosine_similarity(tfidfs)

if __name__ == '__main__':
	corpus = ['Let\'s start with Topic Mining','Bayesian Probability is required for understanding TOpic Mining']
	compute_tfidf_sim(corpus)

