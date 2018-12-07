import bert_sim
import tfidf_sim
import json
import numpy as np
import codecs
import os
from sklearn.metrics.pairwise import cosine_similarity
CORPUS_FILE = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'tmp'),'input2.txt')
NAME_FILE = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'tmp'),'slide_names2.txt')

class Slide_Similarity():
	tfidfs = None
	vec = None
	def __init__(self,courses_json_path=None,sim_on=['titles']):
		if courses_json_path is not None:
			with open(courses_json_path,'r') as f:
				json_data = json.load(f)
			self.build_corpi(json_data,sim_on)
		else:
			with codecs.open(CORPUS_FILE,'r',encoding='utf-8') as f:	
				self.corpus = f.readlines()
			with codecs.open(NAME_FILE,'r',encoding='utf-8') as f:	
				self.slide_names = f.readlines()
			print ('Done loading corpus!')
			self.tfidfs,self.vec = tfidf_sim.build_tfidf(self.corpus)

	def build_corpi(self,json_data,sim_on):
		titles = []
		slides_cnt = []
		subtitles = []
		self.slide_names = []
		lens = []
		for course,lessons in json_data.items():
			for lesson,slides in lessons.items():
				for slide,slide_val in slides.items():
					if 'title' in slide_val.keys():
						titles.append(slide_val['title'])
					else:
						titles.append(' ')
					if 'lecture_transcript' in slide_val.keys():
						subtitles.append(slide_val['lecture_transcript'])
					else:
						subtitles.append(' ')
					if 'text' in slide_val.keys():
						slides_cnt.append(slide_val['text'])
					else:
						slides_cnt.append(' ')
					self.slide_names.append(course+'##'+lesson+'##'+slide)
				#cnt+=cnt1
		self.corpus = []
		if 'titles' in sim_on:
			self.corpus = titles
		if 'content' in sim_on:
			if self.corpus != []:
				self.corpus = ["{} . {}".format(a_.decode(encoding='utf-8',errors='ignore'), b_.decode(encoding='utf-8',errors='ignore')).strip() for a_, b_ in zip(self.corpus, slides_cnt)]
			else:
				self.corpus = slides_cnt
		if 'lecture_transcript' in sim_on:
			if self.corpus != []:
				self.corpus = ["{} . {}".format(a_.decode(encoding='utf-8',errors='ignore'), b_.decode(encoding='utf-8',errors='ignore')).strip() for a_, b_ in zip(self.corpus, subtitles)]
			else:
				self.corpus = subtitles
		self.write_corpus()


	def write_corpus(self):
		with codecs.open(CORPUS_FILE,'w',encoding='utf-8') as f:
			for text in self.corpus:
				f.write(text.replace('\n',' ')+'\n')

		with codecs.open(NAME_FILE,'w',encoding='utf-8') as f:
			for text in self.slide_names:
				f.write(text.replace('\n',' ')+'\n')
		

	def compute_similarity(self,type_='tfidf'):
		if type_ == 'tfidf':
			sim_mat = tfidf_sim.compute_tfidf_sim(self.corpus)
		elif type_ == 'bert':
			sim_mat = bert_sim.compute_bert_sim()
		print (self.corpus[:10])
		print (sim_mat[:10,:10])
		return sim_mat
		

	def get_most_similar_slides(self,out_path,k=10,sim_on=['titles'],type_='tfidf'):
		sim_mat = self.compute_similarity(type_=type_)
		np.fill_diagonal(sim_mat,-1)
		most_sim_argmat = sim_mat.argsort(axis=1)[:,::-1][:,:k]
		most_sim_slides = {}
		for i,slide in enumerate(most_sim_argmat):
			most_sim_slides[self.slide_names[i]] = []
			for sim_slide in slide:
				most_sim_slides[self.slide_names[i]].append({'slide_name':self.slide_names[sim_slide],'sim':sim_mat[i,sim_slide]})
		with open(out_path,'w') as f:
			json.dump(most_sim_slides,f)
		return most_sim_slides

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
	#ss = Slide_Similarity('../courses_json_preprocessed.json',sim_on=['titles','content','lecture_transcript'])
	# print (ss.get_most_similar_slides('sim_slide_bert.json',type_='tfidf',bert_base_dir='/Users/bhavya/Documents/CS510_proj/mooc-web-of-slides/src/slide_similarity/bert_large_uncased'))
	ss = Slide_Similarity()
	sim_mat = ss.compute_similarity(type_= 'tfidf')
	np.save('../data/tfidf-similarity.npy', sim_mat)
	#print (ss.get_most_similar_slides('./results/sim_slide_bert.json',type_='bert'))
