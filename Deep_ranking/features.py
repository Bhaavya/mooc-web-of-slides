import numpy as np
import os
import nltk
import itertools
import io
import csv
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
class Heuristic_features():
	def __init__(self):
		scaler = MinMaxScaler(feature_range=(0.000001,1))
		self.similarity_matrices = []
		self.sim_orders = []
		sim_mats = []
		sim_mats += [np.genfromtxt("../data/average_embeddings_aggregated_matrix.csv",delimiter=',')[:,:-1]]
		sim_mats += [np.load('../data/bert/bert_sim.npy')]
		sim_mats += [np.load('../data/tfidf-similarity.npy')]
		#sim_mats += [np.genfromtxt("../data/word2vec_embeddings.csv",delimiter=',')[:,:-1]]
		sim_mats += [np.load('../data/word2vec_similarities.npy')]		
		sim_mats += [np.genfromtxt("../data/probabilistic_model_matrix.csv",delimiter=',')[:,:-1]]
		sim_mats += [np.load('../data/title_similarity.npy')]
		sim_mats += [np.load('../data/BOW_model_1__100_30_100_100_0.001_ADAM_seq_labels_sim.npy')]
		sim_mats += [np.load('../data/KL_divergence_65_1_0.01_70_sim.npy')]
		sim_mats += [np.load('../data/LDA_similarities_65_topics.npy')]		
		num_embeddings = len(sim_mats)
		for i in range(num_embeddings):
			sim_mat = sim_mats[i]
			sim_mat = np.nan_to_num(sim_mat)
			(x,y) = sim_mat.shape
			sim_mat = scaler.fit_transform(np.reshape(sim_mat, (-1,1)))
			self.similarity_matrices += [np.reshape(sim_mat, (x,y))]

		embeddings_order =  list(csv.reader(open("../data/average_embeddings_aggregated_order.csv", "rb"), delimiter=","))
		self.sim_orders += [[s[0].replace('!!','##') for s in embeddings_order]]
		bert_order = open('../data/bert/slides_names_bert.txt','r').readlines()
		self.sim_orders += [[b.strip() for b in bert_order]]
		tfidf_order = open('../data/slide_names2.txt','r').readlines()
		self.sim_orders += [[b.strip() for b in tfidf_order]]
		word2vec_order = open('../data/slide_names3.txt','r').readlines()
		self.sim_orders += [[b.strip() for b in word2vec_order]]
		#word2vec_embeddings =  list(csv.reader(open("../data/word2vec_embeddings_order.csv", "rb"), delimiter=","))
		#self.sim_orders += [[s[0].replace('!!','##') for s in word2vec_embeddings]]
		probabilistic_model =  list(csv.reader(open("../data/probabilistic_model_order.csv", "rb"), delimiter=","))
		self.sim_orders += [[s[0].replace('!!','##') for s in probabilistic_model]]
		title_order = open('../data/slide_names2.txt','r').readlines()
		self.sim_orders += [[b.strip() for b in title_order]]
		bow_order = open('preprocessed_data/X_names.txt','r').readlines()
		self.sim_orders += [[b.strip() for b in bow_order]]
		KL_order = open('../data/slide_names3.txt','r').readlines()
		self.sim_orders += [[b.strip() for b in KL_order]]
		LDA_order = open('../data/slide_names3.txt','r').readlines()
		self.sim_orders += [[b.strip() for b in LDA_order]]

		correct_order = len(self.sim_orders)-1

		LDA_order_for_bert = ['##'.join(s.split('##')[1:]) for s in self.sim_orders[correct_order]]
		self.similarity_matrices[0] = self.align_order(self.sim_orders[correct_order],self.sim_orders[0],self.similarity_matrices[0])
		self.similarity_matrices[1] = self.align_order(LDA_order_for_bert,self.sim_orders[1],self.similarity_matrices[1])
		for i in range(2,len(self.similarity_matrices)):
			self.similarity_matrices[i] = self.align_order(self.sim_orders[correct_order],self.sim_orders[i],self.similarity_matrices[i])

		print ("Shape: ",self.similarity_matrices[correct_order].shape)

		self.correct_order = correct_order
	def __num_features__(self):
		return len(self.similarity_matrices)

	def align_order(self,order1,order2,order2_mat):
		embeddings_indices = []
		for x in order1:
			embeddings_indices += [order2.index(x)]
		order2_mat = order2_mat[embeddings_indices,:]
		order2_mat = order2_mat[:,embeddings_indices]
		return order2_mat

	def get_features(self,slide1_name,slide2_name):
		x = self.sim_orders[self.correct_order].index(slide1_name)
		y = self.sim_orders[self.correct_order].index(slide2_name)

		features = [sim_mat[x,y] for sim_mat in self.similarity_matrices]

		return features


	def get_deep_embeddings(self,slide1,slide2,slide1_name,slide2_name):
		x = self.sim_orders[0].index(slide1_name)
		y = self.sim_orders[0].index(slide2_name)