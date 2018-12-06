import numpy as np
import json
def is_seq(slide1_name,slide2_name,k=2):
	slide1_name = slide1_name.strip('\n')
	name_comp1 = slide1_name.split('##')
	slide1_number = int(name_comp1[-1].split('slide')[1])
	slide2_name = slide2_name.strip('\n')
	name_comp2 = slide2_name.split('##')
	slide2_number = int(name_comp2[-1].split('slide')[1])
	if name_comp1[:-1]==name_comp2[:-1] and slide1_number!=slide2_number and abs(slide1_number-slide2_number)<=k:
		return True
	else:
		return False

def get_seq_slides_single(slide_name,all_slide_names,k=2):
	slide_name = slide_name.strip('\n')
	name_comp = slide_name.split('##')
	course_name = name_comp[0]
	slide_number = int(name_comp[-1].split('slide')[1])
	seq_slides = []
	for i,cand in enumerate(all_slide_names):
		cand = cand.strip('\n')
		cand_name_comp = cand.split('##')
		cand_course_name = cand_name_comp[0]
		cand_slide_number = int(cand_name_comp[-1].split('slide')[1])
		if name_comp[:-1]==cand_name_comp[:-1] and slide_number!=cand_slide_number and abs(cand_slide_number-slide_number)<=k:
			seq_slides.append(i)
	return seq_slides

def get_seq_slides_all(slide_names,all_slide_names,k=2):
	all_seq_slides = []
	for i,slide_name in enumerate(slide_names):
		all_seq_slides.append(get_seq_slides_single(slide_name,all_slide_names,k))
	# with open('seq_slides.txt','w') as f:
	# 	for i,seq_slide in enumerate(all_seq_slides):
	# 		f.write(slide_names[i]+'\t'+str(seq_slide))
	# 		f.write('\n')
	return all_seq_slides

def calc_recall_k(sim_mat,all_seq_slides,subtitles_lst,k=10):
	np.fill_diagonal(sim_mat,-1)
	k_most_sim = sim_mat.argsort(axis=1)[:,::-1][:,:k]
	total_recall = 0.0
	good_slides = 0.0
	for i,sim_slides in enumerate(k_most_sim):
		# if len(subtitles_lst[i].strip())>0:
			total_recall += len(set(all_seq_slides[i]).intersection(sim_slides))/float(len(all_seq_slides[i]))
			# good_slides+=1
	# print (good_slides)
	return total_recall/(sim_mat.shape[0])

def get_subtitles_lst(slide_names,subtitles_json):
	subtitles_lst = []
	for i,slide_name in enumerate(slide_names):
		subtitles_lst.append(subtitles_json[slide_name.strip('\n')])
	return subtitles_lst

def correct_slide_names(slide_names):
	new_slide_names = []
	for slide_name in slide_names:
		new_slide_names.append(slide_name.replace('!!','##'))
	return new_slide_names

def main(sim_mat_file,slide_names_file,subtitles_file):
	with open(slide_names_file,'r') as f:
		slide_names = f.readlines()
	slide_names = correct_slide_names(slide_names)
	with open(subtitles_file,'r') as f:
		subtitles_json = json.load(f)
	subtitles_lst = get_subtitles_lst(slide_names,subtitles_json)
	all_seq_slides = get_seq_slides_all(slide_names,slide_names)
	sim_mat = np.genfromtxt(sim_mat_file,delimiter=',')
	# sim_mat = np.load(sim_mat_file)
	sim_mat = sim_mat[:,:-1]
	print (sim_mat.shape)
	print (calc_recall_k(sim_mat,all_seq_slides,subtitles_lst))

if __name__ == '__main__':
	main('/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/results/average_embeddings_aggregated_matrix.csv','/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/average_embeddings_aggregated_order.csv','/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/subtitles_wiki.json')
	# print (is_seq('bayesian-methods-in-machine-learning##01_analytical-inference_w1b1.txt##slide0','bayesian-methods-in-machine-learning##01_analytical-inference_w1b1.txt##slide1'))
