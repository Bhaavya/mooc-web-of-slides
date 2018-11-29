import bert_sim
import tfidf_sim
import util

class SlideSimilarity():

	def __init__(self,corpus_file,name_file=None):
		self.corpus = util.read_utf_txt(corpus_file)

	def compute_similarity(self,type_='tfidf',):
		if type_ == 'tfidf':
			sim_mat = tfidf_sim.compute_tfidf_sim(self.corpus)
		elif type_ == 'bert':
			sim_mat = bert_sim.compute_bert_sim()
		print (self.corpus[:10])
		print (sim_mat[:10,:10])
		return sim_mat

if __name__ == '__main__':
	ss = Slide_Similarity('/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/input.txt','/Users/bhavya/Documents/mooc-web-of-slides-local/tmp/src/slide_similarity/slides_names.txt')
	# print (ss.compute_similarity(type_='tfidf'))
