
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
savefile = 'RNN_model_100_15_100_ADAM_seq_labels'

slide_dictionary = np.load('preprocessed_data/slide_dictionary.npy')
vocab_size = len(slide_dictionary)
print ("Vocab size: " , vocab_size)
X = []
with io.open('preprocessed_data/X.txt','r',encoding='utf-8') as f:
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
with io.open('preprocessed_data/X_names.txt','r',encoding='utf-8') as f:
  lines = f.readlines()
for line in lines:
    line = line.strip()
    X_names.append(line)

X_new_names = []
with io.open('../data/slide_names_for_training.txt','r',encoding='utf-8') as f:
  lines = f.readlines()
for line in lines:
    line = line.strip()
    X_new_names.append(line)

indices = []
for s in X_new_names:
  indices += [X_names.index(s)]

X_new = []
for i in indices:
  X_new.append(X[i])

X = X_new
X_names = X_new_names
print(len(X))

batch_size = 100
L_train = len(X_names)
model = torch.load('model/'+savefile+'.model')
model.eval()
q_embeddings = np.zeros((len(X_names), 100),dtype=np.float)
for i in range(0, L_train, batch_size):
  x_input2 = [j for j in X[i:i+batch_size]]
  bs = len(x_input2)
  sequence_length = 100
  x_input = np.zeros((bs,sequence_length),dtype=np.int)
  for j in range(bs):
      x = np.asarray(x_input2[j])
      sl = x.shape[0]
      if(sl < sequence_length):
          x_input[j,0:sl] = x
      else:
          start_index = np.random.randint(sl-sequence_length+1)
          x_input[j,:] = x[start_index:(start_index+sequence_length)]
  q = x_input
  q = torch.LongTensor(q).to(device)
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




