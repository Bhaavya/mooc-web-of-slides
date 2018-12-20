import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import model_zoo
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
import PIL
import random
from torch.nn.modules.loss import _Loss
from models import *
from Triplet_based_ranking import *
import io
import argparse
from features import *
#import sys
#sys.path.insert(0, '/Users/sahiti/Documents/gitlab_repos/mooc-web-of-slides/evaluation/')
import seq_eval


X_names = []
with io.open('preprocessed_data/X_names.txt','r',encoding='utf-8') as f:
  lines = f.readlines()
for line in lines:
    line = line.strip()
    X_names.append(line)

model = torch.load('model/Aggregator_model_100_5_ADAM.model')
savefile = 'Aggregator_model_100_5_ADAM'
heuristic_features = Heuristic_features()
num_features = heuristic_features.__num_features__()
sim_mat = np.zeros((len(X_names),len(X_names)),dtype = np.float)
batch_size = 1000
L_train = len(X_names)
model.eval()
#q_embeddings = np.zeros((len(X_names), 100),dtype=np.float)
for k in range(len(X_names)):
  for i in range(0, L_train, batch_size):
    x_input2 = X_names[i:i+batch_size]
    bs = len(x_input2)
    q = np.zeros((bs,num_features),dtype=np.float)
    #t = np.zeros((bs),dtype = np.float)
    print ("doing")
    for j in range(bs):
      q[j,:] = heuristic_features.get_features(X_names[k],x_input2[j])
    print ("done")
      #t[j] = int(seq_eval.is_seq(x_input2[j][3],x_input2[j][4]) == True) 
    #t = torch.FloatTensor(t).to(device) 
    q = torch.FloatTensor(q).to(device)
    q_output = model(q)
    sim_mat[k,i:i+bs] = q_output.detach().cpu().numpy().reshape((bs))
  print (k)    

np.save('model/'+savefile+"_sim.npy",sim_mat)


#np.save("../data/"+savefile+"_embeddings.npy",q_embeddings)





