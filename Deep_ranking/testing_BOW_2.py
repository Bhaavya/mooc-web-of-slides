
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
import numpy as np
import time
import io
from sklearn import neighbors
from sklearn.metrics.pairwise import cosine_similarity

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print (device)
savefile = 'BOW_model_embeddings_100_30_100_100_0.001_ADAM_seq_labels'

glove_embeddings = np.load('preprocessed_data/glove_embeddings.npy')
vocab_size = 100000
vocab_size += 1
print ("Vocab size: " , vocab_size)
X = []
with io.open('preprocessed_data/X_glove.txt','r',encoding='utf-8') as f:
  lines = f.readlines()
for line in lines:
    line = line.strip()
    if (line != ''):
      line = line.split(' ')
      line = np.asarray(line,dtype=np.int) 
      X.append(line)
    else:
      X.append(np.asarray([]))

X_names = []
with io.open('preprocessed_data/X_names_glove.txt','r',encoding='utf-8') as f:
  lines = f.readlines()
for line in lines:
    line = line.strip()
    X_names.append(line)
batch_size = 100
L_train = len(X_names)
model = torch.load('model/'+savefile+'.model')
model.eval()
q_embeddings = np.zeros((len(X_names), 100),dtype=np.float)
for i in range(0, L_train, batch_size):
  x_input2 = [np.mean(glove_embeddings[j],axis=0) for j in X[i:i+batch_size]]
  bs = len(x_input2)
  q = torch.FloatTensor(x_input2).to(device)
  q_embeddings[i:i+bs,:] = model(q).detach().cpu().numpy()
  if (i+1)%1 == 0:
    print (i)
deep_rank_matrix = cosine_similarity(q_embeddings)

np.save("../data/"+savefile+"_sim.npy",deep_rank_matrix)

np.save("../data/"+savefile+"_embeddings.npy",q_embeddings)


target = open("Deep_ranking_results.txt","w")
x = np.random.choice(len(X_names), size=200)
sample = deep_rank_matrix[x,:]
indices = np.argsort(sample,axis=1)[:,::-1][:,:10]
X_names = np.array(X_names)
for i in range(200):
  target.write(str(X_names[x[i]])+'\n')
  target.write("similar"+'\n')
  target.write(str(zip(list(X_names[indices[i,:]]), list(deep_rank_matrix[x[i],indices[i,:]])))+'\n')
target.flush()
target.close()




