from sklearn.feature_extraction.text import TfidfVectorizer as tv
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf(corpus):
	tfidfs = tv(corpus)
	return tfidfs

def compute_tfidf_sim(corpus):
	tfidfs = build_tfidf(corpus)
	return cosine_similarity(tfidfs)

