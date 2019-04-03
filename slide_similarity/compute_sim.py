import bert_sim
import tfidf_sim
import util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
class SlideSimilarity():

	def __init__(self,corpus_file,sim_mat_file=None):
		self.corpus_file = corpus_file
		self.sim_mat_file = sim_mat_file
		self.corpus = util.read_utf_txt(self.corpus_file)


	def compute_similarity(self,type_='tfidf'):
		if type_ == 'tfidf':
			corpus = util.read_utf_txt(self.corpus_file)
			sim_mat = tfidf_sim.compute_tfidf_sim(corpus)
		elif type_ == 'bert':
			sim_mat = bert_sim.compute_bert_sim(self.corpus_file)
		print (sim_mat[:10,:50])
		if self.sim_mat_file is not None:
			np.save(self.sim_mat_file,sim_mat)
		return sim_mat

	def get_similar_slides(self,text,k=10):
		#vocabulary = 
		text_V = self.vec.transform([text])
		sim = cosine_similarity(text_V,self.tfidfs)
		#print (sim.shape)
		arg_sim_slides = sim.argsort()[0][::-1][:k]
		most_sim_slides = []
		#print arg_sim_slides
		for slide in arg_sim_slides:
			#print (slide)
			#print self.slide_names[slide]
			most_sim_slides += [(self.slide_names[slide],sim[0,slide])]
		return most_sim_slides

if __name__ == '__main__':
	#ss = SlideSimilarity('/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/input_bert.txt',sim_mat_file='/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/results/tfidf_sim.npy').compute_similarity('tfidf')
	ss = SlideSimilarity('tmp/input2.txt')
	tfidfs,vec = tfidf_sim.build_tfidf(ss.corpus) 
	only_titles = [s.split(" . ")[0] for s in ss.corpus]
	print only_titles
	title_tfidfs = vec.transform(only_titles)
	sim_mat = cosine_similarity(title_tfidfs)
	np.save("../data/title_similarity.npy",sim_mat)


