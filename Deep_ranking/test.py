import numpy as np
import csv
def align_order(order1,order2,order2_mat):
	embeddings_indices = []
	for x in order1:
		embeddings_indices += [order2.index(x)]
	order2_mat = order2_mat[embeddings_indices,:]
	order2_mat = order2_mat[:,embeddings_indices]
	return order2_mat
sim_orders = []
sim_mats = [np.genfromtxt("../data/average_embeddings_aggregated_matrix.csv",delimiter=',')[:,:-1]]	
probabilistic_model =  list(csv.reader(open("../data/average_embeddings_aggregated_order.csv", "rb"), delimiter=","))
sim_orders += [[s[0].replace('!!','##') for s in probabilistic_model]]
word2vec_order = open('../data/slide_names_for_training.txt','r').readlines()
sim_orders += [[b.strip() for b in word2vec_order]]
#LDA_order_for_bert = ['##'.join(s.split('##')[1:]) for s in sim_orders[1]]
sim_mats = align_order(sim_orders[1],sim_orders[0],sim_mats[0])
np.save("../data/average_embeddings_aggregated_matrix_2.npy", sim_mats)
