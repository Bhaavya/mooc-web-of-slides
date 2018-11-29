import codecs
import json
import numpy as np 


def write_utf_txt(corpus,corpus_file):
	with codecs.open(corpus_file,'w',encoding='utf-8') as f:
			for text in corpus:
				f.write(text.replace('\n',' ')+'\n')

def read_utf_txt(corpus_file):
	with codecs.open(corpus_file,'r',encoding='utf-8') as f:	
				return f.readlines()

def get_top_slides(score_mat,out_path,slide_names,k=10,addn_info_keys_mat=None,addn_info_mat=None):
		np.fill_diagonal(score_mat,-1)
		most_sim_argmat = score_mat.argsort(axis=1)[:,::-1][:,:k]
		most_sim_slides = {}
		for i,slide in enumerate(most_sim_argmat):
			most_sim_slides[slide_names[i]] = []
			for sim_slide in slide:
				sim_dict = {'slide_name':slide_names[sim_slide],'sim':score_mat[i,sim_slide]}
				for j,key in enumerate(addn_info_keys_mat):
					sim_dict[key] = addn_info_mat[j][i,sim_slide]
				most_sim_slides[slide_names[i]].append(sim_dict)
		with open(out_path,'w') as f:
			json.dump(most_sim_slides,f)
		return most_sim_slides
