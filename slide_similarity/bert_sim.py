from subprocess import call 
import os 
import shlex
import json
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import codecs



def run_extract_features(bert_base_dir,input_file,output_file,seq_length=512,batch_size=1024):
	cmd = 'python {} --input_file={} --output_file={} --vocab_file={}vocab.txt --bert_config_file={}bert_config.json --init_checkpoint={}bert_model.ckpt --layers=-1,-2,-3,-4 --max_seq_length={} --batch_size={}'.format(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'bert'),'extract_features.py'),input_file,output_file,bert_base_dir+os.sep,bert_base_dir+os.sep,bert_base_dir+os.sep,seq_length,batch_size)

	call(shlex.split(cmd))

def get_corpus_embeddings(bert_wts_path):
	doc_embeddings = []
	with open(bert_wts_path,'r') as f:
		wts = f.readlines()
	print ('Done loading features!')
	for wt in wts:
		json_wt = json.loads(wt)
		doc_wt = json_wt['features'][0]
		doc_embedding = []
		if doc_wt['token'] == "[CLS]":
			for layer in doc_wt['layers']:
				doc_embedding += layer['values']
			doc_embeddings.append(doc_embedding)
		else:
			print ("Warning: CLS is not the first token!!!\n")
	del wts 
	return np.array(doc_embeddings)

def compute_bert_sim(bert_wts_path):
	bert_embeddings = get_corpus_embeddings(bert_wts_path)
	return cosine_similarity(bert_embeddings)

if __name__ == '__main__':
	bert_base_dir = '/Users/bhavya/Documents/CS510_proj/mooc-web-of-slides/src/slide_similarity/bert_large_uncased'
	input_file = '/Users/bhavya/Documents/CS510_proj/mooc-web-of-slides/src/slide_similarity/tmp/input.txt'
	output_file = '/Users/bhavya/Documents/CS510_proj/mooc-web-of-slides/src/slide_similarity/tmp/output.jsonl'
	run_extract_features(bert_base_dir)