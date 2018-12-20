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
'''
slide_dictionary = np.load('preprocessed_data/slide_dictionary.npy')
vocab_size = len(slide_dictionary)
print ("Vocab size: " , vocab_size)
X = []
with io.open('preprocessed_data/X.txt','r',encoding='utf-8') as f:
  lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int) 
    X.append(line)
'''
X_names = []
with io.open('../data/slide_names_for_training.txt','r',encoding='utf-8') as f:
  lines = f.readlines()
for line in lines:
    line = line.strip()
    X_names.append(line)


heuristic_features = Heuristic_features()
num_features = heuristic_features.__num_features__()
print (num_features)


savefile = 'Logistic_regression_'+str(num_features)+'_BOW_included'

model = pickle.load(open('model/'+savefile+'.sav', 'rb'))

n1 = 388
n=len(X_names)
sim_mat = np.zeros((n1,n),dtype=np.float)
total_acc = 0
start = 2000
for i in range(n1):
  X_test = []
  Y_test = []
  for j in range(n):
    X_test.append(heuristic_features.get_features(X_names[start+i],X_names[j]))
    Y_test.append(int(seq_eval.is_seq(X_names[start+i],X_names[j])==True))
  running_acc = model.score(X_test,Y_test)
  s = model.predict_proba(np.array(X_test))
  sim_mat[i,:] = s[:,list(model.classes_).index(1)]
  total_acc += running_acc
  if(i%20==0):
    print (start+i)
    print ("running accuracy:",(total_acc/float(i+1)))
print ("Total accuracy:", float(total_acc)/float(1000))

np.save('logreg_results/'+savefile+"_2000_2388.npy",sim_mat)