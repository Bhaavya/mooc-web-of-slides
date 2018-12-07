
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
from sklearn import neighbors
from sklearn.metrics.pairwise import cosine_similarity

# Device configuration
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
print (device)


slide_dictionary = np.load('preprocessed_data/slide_dictionary.npy.npy')
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
with io.open('preprocessed_data/X_names.txt','r',encoding='utf-8') as f:
  lines = f.readlines()
for line in lines:
    line = line.strip()
    X_names.append(line)


model = torch.load('model/'+savefile+'.model')

q_embeddings = np.zeroes((X.size().numpy()))
for i, (q_image) in enumerate(X):
  q_image = q_image.to(device)
  print (model(q_image).size())
  q_embeddings[i,:] = model(q_image).cpu().numpy()

deep_rank_matrix = cosine_similarity(q_embeddings)





