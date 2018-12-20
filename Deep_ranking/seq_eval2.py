import numpy as np
import json
import csv
import re

IGNORE_LEC = ['text-mining-analytics##06_topic-modeling##01_topic-modeling-with-mallets-lda-and-dmr##06_introduction-to-probabilistic-topic-models-by-david-blei_WangBlei2011.txt',
'cs-410##01_orientation##01_orientation-information##03_syllabus_CSGraduateStudentHandbook_web.15-16.txt','_Han_Data_Mining_3e_Chapters_2101113']

def is_seq(slide1_name,slide2_name,m):
	slide1_name = slide1_name.strip('\n')
	name_comp1 = slide1_name.split('##')
	slide1_number = int(name_comp1[-1].split('slide')[1])
	slide2_name = slide2_name.strip('\n')
	name_comp2 = slide2_name.split('##')
	slide2_number = int(name_comp2[-1].split('slide')[1])
	if name_comp1[:-1]==name_comp2[:-1] and slide1_number!=slide2_number and abs(slide1_number-slide2_number)<=m:
		return True
	else:
		return False

def get_seq_slides_single(slide_name,all_slide_names,m):
	seq_slides = []
	for i,cand in enumerate(all_slide_names):
		if is_seq(slide_name,cand,m):
			seq_slides.append(i)
	return seq_slides

def get_seq_slides_all(slide_names_rows,slide_names_cols,m=2):
	all_seq_slides = []
	for i,slide_name in enumerate(slide_names_rows):
		all_seq_slides.append(get_seq_slides_single(slide_name,slide_names_cols,m))
	# with open('seq_slides.txt','w') as f:
	# 	for i,seq_slide in enumerate(all_seq_slides):
	# 		f.write(slide_names[i]+'\t'+str(seq_slide))
	# 		f.write('\n')
	return all_seq_slides

def is_actual_slide(i,slide_names,slide_info_lst):
	f = True
	for ig_lec in IGNORE_LEC:
		if ig_lec in slide_names[i]:
			f=False
	if slide_info_lst[i]['num_words']==0:
		f=False
	return f

def calc_recall_k(sim_mat,all_seq_slides,subtitles_lst,slide_rows_info_lst,slide_cols_info_lst,slide_names_rows,slide_names_cols,k=10,analysis_file_prefix=None):
	np.fill_diagonal(sim_mat,-10000)
	most_sim = sim_mat.argsort(axis=1)[:,::-1]
	print ('Sorted sim mat:\n',np.sort(sim_mat,axis=1)[:,::-1][:,:k])
	total_recall = 0.0
	good_slides_recall = 0.0
	good_slides = 0.0
	actual_slides = 0.0
	k_most_sim_mat= []
	for i,sim_slides in enumerate(most_sim):
		k_most_sim_mat.append([])

		if is_actual_slide(i,slide_names_rows,slide_rows_info_lst):
			actual_slides+=1
			k_most_sim_mat[i] = ['None' for s in range(k)]
			sim_slide_num = 0
			for j in sim_slides:
				if not sim_slide_num==k:
					if is_actual_slide(j,slide_names_cols,slide_cols_info_lst):
						k_most_sim_mat[i][sim_slide_num] =j
						sim_slide_num+=1
				else:
					break
			total_recall += len(set(all_seq_slides[i]).intersection(k_most_sim_mat[i]))/float(len(all_seq_slides[i]))
			if len(subtitles_lst[i].strip())>0:
				good_slides_recall += len(set(all_seq_slides[i]).intersection(k_most_sim_mat[i]))/float(len(all_seq_slides[i]))
				good_slides+=1

	print ('\n','-'*20,'Actual Slides:',actual_slides,'-'*20)
	print ('\n','-'*20,'Slides with subtitles',good_slides,'-'*20)
	print ('\n','-'*20, 'Recall All Slides:',total_recall/(actual_slides),'-'*20)
	print ('\n','-'*20, 'Recall Subtitles Slides:',good_slides_recall/(good_slides),'-'*20)
	if analysis_file_prefix is not None:
		analyze_results(k_most_sim_mat,sim_mat,all_seq_slides,analysis_file_prefix,slide_names_rows,slide_names_cols,slide_rows_info_lst,slide_cols_info_lst)

def analyze_results(k_most_sim,sim_mat,all_seq_slides,analysis_file_prefix,slide_names_rows,slide_names_cols,slide_rows_info_lst,slide_cols_info_lst):
	failures = []
	winners = []
	print (k_most_sim.shape)
	for i,sim_slides in enumerate(k_most_sim):
		if sim_slides[0]!='None':
			curr_slide_info = slide_rows_info_lst[i]
			for j in sim_slides:
				sim_slide_info = slide_cols_info_lst[j]
				info = {'name':slide_names_rows[i],'k_most_sim':slide_names_cols[j],'similarity':sim_mat[i,j]}
				for k,v in curr_slide_info.items():
					info[k+'_slide'] = v
				for k,v in sim_slide_info.items():
					info[k+'_sim_slide'] = v

				if len(set(all_seq_slides[i]).intersection(sim_slides)) == 0:
					failures.append(info)
				else:
					winners.append(info)

	keys = failures[0].keys()
	with open(analysis_file_prefix+'_failures.csv', 'w') as output_file:
		dict_writer = csv.DictWriter(output_file, keys)
		dict_writer.writeheader()
		dict_writer.writerows(failures)

	keys = winners[0].keys()
	with open(analysis_file_prefix+'_winners.csv', 'w') as output_file:
		dict_writer = csv.DictWriter(output_file, keys)
		dict_writer.writeheader()
		dict_writer.writerows(winners)

def analyze_results2(k_most_sim_mat,sim_mat,all_seq_slides,analysis_file_prefix,slide_names_rows,slide_names_cols,slide_rows_info_lst,slide_cols_info_lst):
	failures = {}
	winners = {}
	print (k_most_sim.shape)
	for i,sim_slides in enumerate(k_most_sim):
		if sim_slides[0]!='None':
			slide_dict = {}
			slide_dict[slide_names_rows[i]] = []
			for j in sim_slides:
					slide_dict[slide_names_rows[i]].append(slide_names_cols[j])
			if len(set(all_seq_slides[i]).intersection(sim_slides)) == 0:
				failures.update(slide_dict)
			else:
				winners.update(slide_dict)


	with open(analysis_file_prefix+'_failures.txt', 'w') as output_file:
		for k,v in failures.items():
			output_file.write(k)
			output_file.write('similar'+'\n'+'[')
			for x in v:
				output_file.write(x)
			output_file.write(']'+'\n')


	with open(analysis_file_prefix+'_winners.txt', 'w') as output_file:
		for k,v in winners.items():
			output_file.write(k)
			output_file.write('similar'+'\n'+'[')
			for x in v:
				output_file.write(x)
			output_file.write(']'+'\n')

		
def get_subtitles_lst(slide_names_rows,subtitles_json):
	subtitles_lst = []
	for i,slide_name in enumerate(slide_names_rows):
		subtitles_lst.append(subtitles_json[slide_name.strip('\n')])
	return subtitles_lst

def correct_slide_names(slide_names):
	new_slide_names = []
	for slide_name in slide_names:
		new_slide_names.append(slide_name.replace('!!','##'))
	return new_slide_names

def get_slide_info(courses_json,all_slide_names):
	with open(courses_json,'r') as f:
		json_data = json.load(f)
	slide_info_dict ={}
	for course,lessons in json_data.items():
		for lesson,slides in lessons.items():
			for slide,slide_val in slides.items():
				slide_name = course+'##'+lesson+'##'+slide+'\n'
				text =''
				try:
					text += slide_val['title'] +' '
				except:
					pass
				try:
					text += slide_val['lecture_transcript'] +' '
				except:
					pass
				try:
					text += slide_val['text'] +' '
				except:
					pass
				words = text.split()
				num_words = len(words)
				num_symbols = len([w for w in words if len(w)<2 and w not in [',',' ','.','"',"'",'-'] and not re.match('\d',w)])
				slide_info_dict[slide_name] = {'num_words':num_words,'num_symbols':num_symbols}
	slide_info_lst = []
	for slide_name in all_slide_names:
		slide_info_lst.append(slide_info_dict[slide_name])
	return slide_info_lst


def main(sim_mat_file,slide_names_rows_file,slide_names_cols_file,subtitles_file,format,correct_names,analysis_file_prefix=None,courses_json=None):
	with open(slide_names_rows_file,'r') as f:
		slide_names_rows = f.readlines()
	with open(slide_names_cols_file,'r') as f:
		slide_names_cols = f.readlines()

	if correct_names.lower() == 'y':
		slide_names_cols = correct_slide_names(slide_names_cols)
		slide_names_rows = correct_slide_names(slide_names_rows)

	print ('Number of slide names:',len(slide_names_rows),len(slide_names_cols))

	with open(subtitles_file,'r') as f:
		subtitles_json = json.load(f)

	subtitles_lst = get_subtitles_lst(slide_names_rows,subtitles_json)

	all_seq_slides = get_seq_slides_all(slide_names_rows,slide_names_cols)

	if format == 'csv':
		sim_mat = np.genfromtxt(sim_mat_file,delimiter=',')
		sim_mat = sim_mat[:,:-1]

	elif format =='npy':
		sim_mat = np.load(sim_mat_file)

	sim_mat = np.nan_to_num(sim_mat)

	print ("Max and min similarity:",np.max(sim_mat),np.min(sim_mat))
	print ('Sim mat shape',sim_mat.shape)
	print ('Sim mat sample:\n',sim_mat[:10,:10])

	slide_rows_info_lst = get_slide_info(courses_json,slide_names_rows)
	slide_cols_info_lst = get_slide_info(courses_json,slide_names_cols)

	
	calc_recall_k(sim_mat,all_seq_slides,subtitles_lst,slide_rows_info_lst,slide_cols_info_lst,slide_names_rows,slide_names_cols,analysis_file_prefix=analysis_file_prefix)

if __name__ == '__main__':
	# main('/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/results/aggregation_matrix1.npy','/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/average_embeddings_aggregated_order.csv','/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/subtitles_wiki.json')
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--format', type=str, default='npy', help='npy, csv')
    parser.add_argument('--mat', type=str, help='path to similarity matrix')
    parser.add_argument('--row_names', type=str, help='path to row slides names order')
    parser.add_argument('--col_names', type=str, help='path to col slides names order')
    parser.add_argument('--subtitles', type=str, help='path to subtitles')
    parser.add_argument('--correct_names', type=str, help='replace !! with ##')
    parser.add_argument('--analysis_file_prefix', type=str, help='prefix to result analysis file',default=None)
    parser.add_argument('--courses_json', type=str, help='complete slides info info',default=None)


    args = parser.parse_args()

    main(args.mat,args.row_names,args.col_names,args.subtitles,args.format,args.correct_names,args.analysis_file_prefix,args.courses_json)

