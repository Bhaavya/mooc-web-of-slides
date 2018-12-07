
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
from preprocess_data import preprocess_matrices
from preprocess_data import combine_matrices
class triplettrainDataset(Dataset):
    """Face Landmarks dataset."""
    class_names = []
    def __init__(self, x_train, x_names, alphas, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.alphas = alphas
        s = preprocess_matrices()
        self.similarity_matrix,self.dissimilarity_matrix= combine_matrices(alphas,[s[0],s[1]])
        self.embeddings_order = s[2]
        self.x_train = x_train
        self.x_names = x_names
    def __len__(self):
        return len(self.x_names)

    def __getitem__(self, idx):
      slide_name = self.x_names[idx]
      order_idx = self.embeddings_order.index(slide_name) 
      sim_probabilities = self.similarity_matrix[order_idx,:]
      dissim_probabilities = self.dissimilarity_matrix[order_idx,:]
      positive_choice = np.random.choice(np.arange(0, len(self.x_names)), p=sim_probabilities)
      negative_choice = np.random.choice(np.arange(0, len(self.x_names)), p=dissim_probabilities)
      pos_slide_name = self.embeddings_order[positive_choice]
      neg_slide_name = self.embeddings_order[negative_choice]
      p_idx = self.x_names.index(pos_slide_name)
      n_idx = self.x_names.index(neg_slide_name)
      q = self.x_train[idx]
      p = self.x_train[p_idx]
      n = self.x_train[n_idx]
      return (q,p,n)

class testDataset(Dataset):
  def __init__(self, x_test, x_names, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x_test = x_test
        self.x_names = x_names
  def __len__(self):
        return len(self.x_names)

  def __getitem__(self, idx):
      return (x_test[idx])

class Similarityloss(_Loss):
  def __init__(self, gap = 1.0, zero = 0.0):
    super(Similarityloss, self).__init__()
    self.pdist = nn.CosineSimilarity()
    self.gap = gap
  def forward(self, zero, query, positive, negative):
    result = torch.max(zero, self.gap-self.pdist(query, positive)+self.pdist(query,negative))
    return result.mean()

class triplettrainDataset_aggregator(Dataset):
    """Face Landmarks dataset."""
    class_names = []
    def __init__(self, x_train, x_names, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.alphas = alphas
        #s = preprocess_matrices()
        #self.similarity_matrix,self.dissimilarity_matrix= combine_matrices(alphas,[s[0],s[1]])
        #self.embeddings_order = s[2]
        self.x_train = x_train
        self.x_names = x_names
    def __len__(self):
        return len(self.x_names)

    def __getitem__(self, idx):
      slide_name = x_names[idx]
      s1_name = random.choice(list(set(x_names)-set(slide_name)))
      s2_name = random.choice(list(set(x_names)-set(slide_name,s1)))
      s1_idx = x_names.index(s1_name)
      s2_idx = x_names.index(s2_name)
      q = x_train[idx]
      s1 = x_train[s1_idx]
      s2 = x_train[s2_idx]
      return (q,s1,s2, slide_name, s1_name, s2_name)

class Similarityloss(_Loss):
  def __init__(self, gap = 1.0, zero = 0.0):
    super(Similarityloss, self).__init__()
    self.pdist = nn.CosineSimilarity()
    self.gap = gap
  def forward(self, zero, query, positive, negative):
    result = torch.max(zero, self.gap-self.pdist(query, positive)+self.pdist(query,negative))
    return result.mean()


