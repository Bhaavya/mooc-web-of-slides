from compute_sim import SlideSimilarity
import numpy as np 
import util

def find_alt_exp(title_path,content_path,slide_names_path,out_path,alpha=1):
	ss1 = SlideSimilarity(title_path)
	ss2 = SlideSimilarity(content_path)
	title_sim = ss1.compute_similarity()
	content_sim = ss2.compute_similarity()
	alt_exp_score = np.divide(alpha*title_sim, content_sim, out=alpha*title_sim, where=content_sim!=0)
	slide_names = util.read_utf_txt(slide_names_path)
	util.get_top_slides(alt_exp_score,out_path,slide_names,addn_info_keys_mat=['title_sim','content_sim'],addn_info_mat=[title_sim,content_sim])

if __name__ == '__main__':
	find_alt_exp('/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/titles.txt','/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/concat.txt','/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/slides_names.txt','/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/results/alt_exp_title_cntsub.json')


