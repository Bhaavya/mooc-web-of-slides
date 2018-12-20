from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import PIL
import random
from models import *
from Triplet_based_ranking import *
import io
import argparse
from features import *
import sys
#sys.path.insert(0, '/Users/sahiti/Documents/gitlab_repos/mooc-web-of-slides/evaluation/')
import seq_eval
import time
import pickle

slide_dictionary = np.load('preprocessed_data/slide_dictionary.npy')
vocab_size = len(slide_dictionary)
print ("Vocab size: " , vocab_size)

X_names = []
with io.open('../data/slide_names_for_training.txt','r',encoding='utf-8') as f:
  lines = f.readlines()
for line in lines:
    line = line.strip()
    X_names.append(line)


#five-fold division
fivefold_test = [range(i,(i+int(len(X_names)/5))) for i in range(0, len(X_names), int(len(X_names)/5))] 
s = range(len(X_names))
fivefold_train = [list(set(s)-set(x)) for x in fivefold_test]
I_permuatation = np.random.permutation(range(len(X_names)))
fivefold_train_names = [[X_names[I_permuatation[l]] for l in fivefold_train[0]]]
fivefold_test_names = [[X_names[I_permuatation[l]] for l in  fivefold_test[0]]]
x_train_names = fivefold_train_names[0]
x_test_names = fivefold_test_names[0]
#x_val_names = x_test_names[:int(len(x_test_names)/2)]
#x_test_names = x_test_names[int(len(x_test_names)/2):]
## Parser
print ("Total: ", len(X_names))
print ("Training ", len(x_train_names))

heuristic_features = Heuristic_features()
num_features = heuristic_features.__num_features__()
print (num_features)
n = len(x_train_names)
savefile = '_'.join(['Logistic_regression', str(num_features), 'BOW_included'])
print (savefile)
with io.open('logreg_results/'+'X_'+savefile+'_test_names.txt','w',encoding='utf-8') as f:
  for name in x_test_names:
    f.write(name)
    f.write(("\n").encode("utf-8").decode("utf-8"))


prob_matrix = np.zeros((n,n),dtype = np.float)
y_train = np.zeros((n,n),dtype = np.float)
for i in range(n):
  sum_ones = 0
  for j in range(n):
    prob_matrix[i][j] = int(seq_eval.is_seq(x_train_names[i],x_train_names[j])==True)
    y_train[i][j] = prob_matrix[i][j]
    sum_ones += prob_matrix[i][j]
  if(sum_ones ==0):
    print (x_train_names[i])
    sum_ones = int(n/2)
  num_zeros = n-1-sum_ones 
  prob_matrix[i][prob_matrix[i]==1] = float(0.5)/float(sum_ones)
  prob_matrix[i][prob_matrix[i]==0] = float(0.5)/float(num_zeros)
  prob_matrix[i][i] = 0


X_train = []
Y_train = []
for i in range(n):
	sample = np.random.choice(n, size=200,replace=False,p=prob_matrix[i])
	for j in sample:
		X_train.append(heuristic_features.get_features(x_train_names[i],x_train_names[j]))
		Y_train.append(y_train[i][j])
	if(i%100==0):
		print (i)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print (X_train.shape)
print (Y_train.shape)
time1 = time.time()
model = LogisticRegression(solver = 'sag').fit(X_train,Y_train)
time2=time.time()
print (time2-time1)
score = model.score(X_train,Y_train)
print ("Training Accuracy:",score)
print (model.classes_)
pickle.dump(model, open('model/'+savefile+'.sav', 'wb'))



n1 = 1000
n2 = len(X_names)
sim_mat = np.zeros((n1,n2),dtype=np.float)
total_acc = 0
for i in range(n1):
  X_test = []
  Y_test = []
  for j in range(n2):
    X_test.append(heuristic_features.get_features(X_names[i],X_names[j]))
    Y_test.append(int(seq_eval.is_seq(X_names[i],X_names[j])==True))
  running_acc = model.score(X_test,Y_test)
  s = model.predict_proba(np.array(X_test))
  sim_mat[i,:] = s[:,list(model.classes_).index(1)]
  total_acc += running_acc
  if(i%20==0):
    print (i)
    print ("running accuracy:",(total_acc/float(i+1)))
  
print ("Total accuracy:", float(total_acc)/float(1000))
print (sim_mat.shape)
np.save('logreg_results/'+savefile+"_similarity_1000.npy",sim_mat)




print (len(x_test_names))
n1 = len(x_test_names)
n2 = len(X_names)
sim_mat = np.zeros((n1,n2),dtype=np.float)
total_acc = 0
for i in range(n1):
	X_test = []
	Y_test = []
	for j in range(n2):
		X_test.append(heuristic_features.get_features(x_test_names[i],X_names[j]))
		Y_test.append(int(seq_eval.is_seq(x_test_names[i],X_names[j])==True))
	running_acc = model.score(X_test,Y_test)
	s = model.predict_proba(np.array(X_test))
	sim_mat[i,:] = s[:,list(model.classes_).index(1)]
	total_acc += running_acc
	if(i%20==0):
		print (i)
		print ("running accuracy:",(total_acc/float(i+1)))
	
print ("Total accuracy:", float(total_acc)/float(1000))
print (sim_mat.shape)
np.save('logreg_results/'+savefile+"_similarity_test2.npy",sim_mat)

x_test_indices = []
for name in x_test_names:
  x_test_indices += [X_names.index(name)]
sim_mat = sim_mat[:,x_test_indices]
print (sim_mat.shape)
np.save('logreg_results/'+savefile+ "_similarity_test.npy",sim_mat)


