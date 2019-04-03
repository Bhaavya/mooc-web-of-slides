
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
import sys
#sys.path.insert(0, '/Users/sahiti/Documents/gitlab_repos/mooc-web-of-slides/evaluation/')
import seq_eval
from sklearn import preprocessing
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
        #s = preprocess_matrices()
        #self.similarity_matrix,self.dissimilarity_matrix= combine_matrices(alphas,s[0])
        #self.embeddings_order = s[1]
        #for j in range(len(x_names)):
        #  required_indices += embeddings_order.index(x_names[j])

        prob_matrix = np.zeros((len(x_names),len(x_names)),dtype = np.float)
        for i in range(len(x_names)):
          sum_ones = 0
          for j in range(len(x_names)):
            prob_matrix[i][j] = int(seq_eval.is_seq(x_names[i],x_names[j])==True)
            sum_ones += prob_matrix[i][j]
          if(sum_ones ==0):
            print (x_names[i])
            sum_ones = int(len(x_names)/2)
          num_zeros = len(x_names)-1-sum_ones 
          prob_matrix[i][prob_matrix[i]==1] = float(0.9)/float(sum_ones)
          prob_matrix[i][prob_matrix[i]==0] = float(0.1)/float(num_zeros)
          prob_matrix[i][i] = 0
        self.prob1_matrix = prob_matrix
        prob_matrix[prob_matrix==0] =  0.00000001
        prob2_matrix = float(1)/prob_matrix
        np.fill_diagonal(prob2_matrix, 0)
        self.prob1_matrix = preprocessing.normalize(prob_matrix,norm = 'l1')
        self.prob2_matrix = preprocessing.normalize(prob2_matrix,norm = 'l1')
        
        self.x_train = x_train
        self.x_names = x_names
    def __len__(self):
        return len(self.x_names)

    def __getitem__(self, idx):
      slide_name = self.x_names[idx]
      
      #order_idx = self.embeddings_order.index(slide_name) 
      #sim_probabilities = self.similarity_matrix[order_idx,:]
      #dissim_probabilities = self.dissimilarity_matrix[order_idx,:]
      #positive_choice = np.random.choice(np.arange(0, len(self.x_names)), p=sim_probabilities)
      #negative_choice = np.random.choice(np.arange(0, len(self.x_names)), p=dissim_probabilities)
      positive_choice = np.random.choice(np.arange(0, len(self.x_names)), p=self.prob1_matrix[idx])
      negative_choice = np.random.choice(np.arange(0, len(self.x_names)), p=self.prob2_matrix[idx])
      pos_slide_name = self.x_names[positive_choice] #self.embeddings_order[positive_choice]
      neg_slide_name = self.x_names[negative_choice] #self.embeddings_order[negative_choice]
      #pos_slide_name = self.embeddings_order[positive_choice]
      #neg_slide_name = self.embeddings_order[negative_choice]
      p_idx = self.x_names.index(pos_slide_name)
      n_idx = self.x_names.index(neg_slide_name)
      q = self.x_train[idx]
      p = self.x_train[p_idx]
      n = self.x_train[n_idx]
      return (q,p,n,slide_name,pos_slide_name,neg_slide_name)

class triplettrainDataset2(Dataset):
  class_names = []
  def __init__(self, X, X_names, x_train_names, x_train_pairs, y_train):
    index = 0
    print (x_train_pairs[:3])
    x_train_dict_pos_examples = {}
    x_train_dict_neg_examples = {}
    for name in x_train_names:
      x_train_dict_pos_examples[name] = []
      x_train_dict_neg_examples[name] = []
    for (name1,name2) in x_train_pairs:
      if y_train[index] == 1:
        x_train_dict_pos_examples[name1] += [name2]
      else:
        x_train_dict_neg_examples[name1] += [name2]
      index += 1
    self.x_train_dict_pos_examples = x_train_dict_pos_examples
    self.x_train_dict_neg_examples = x_train_dict_neg_examples
    self.x_names = X_names
    self.x_train = X
    self.x_train_names = x_train_names
  def __getitem__(self, idx):
    slide_name = self.x_train_names[idx]
    p_name = np.random.choice(self.x_train_dict_pos_examples[slide_name],1)
    n_name = np.random.choice(self.x_train_dict_neg_examples[slide_name],1)
    p_idx = self.x_names.index(p_name[0])
    n_idx = self.x_names.index(n_name[0])
    idx = self.x_names.index(slide_name)
    q = self.x_train[idx]
    p = self.x_train[p_idx]
    n = self.x_train[n_idx]
    return (q,p,n,slide_name,p_name[0],n_name[0])
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
    result = torch.max(zero, self.gap-(self.pdist(query, positive))+(self.pdist(query,negative)))
    return result.mean()

class Similarityloss2(_Loss):
  def __init__(self, gap = 1.0, zero = 0.0):
    super(Similarityloss, self).__init__()
    self.pdist = nn.CosineSimilarity()
    self.gap = gap
  def forward(self, zero, query, positive, negative):
    result = torch.max(zero, self.gap-(self.pdist(query, positive))+(self.pdist(query,negative)))
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
        prob_matrix = np.zeros((len(x_names),len(x_names)),dtype = np.float)
        for i in range(len(x_names)):
          sum_ones = 0
          for j in range(len(x_names)):
            prob_matrix[i][j] = int(seq_eval.is_seq(x_names[i],x_names[j])==True)
            sum_ones += prob_matrix[i][j]
          if(sum_ones ==0):
            print (x_names[i])
            sum_ones = int(len(x_names)/2)
          num_zeros = len(x_names)-1-sum_ones 
          prob_matrix[i][prob_matrix[i]==1] = float(0.5)/float(sum_ones)
          prob_matrix[i][prob_matrix[i]==0] = float(0.5)/float(num_zeros)
          prob_matrix[i][i] = 0
        self.prob_matrix = prob_matrix

    def __len__(self):
        return len(self.x_names)

    def __getitem__(self, idx):
      x_names = self.x_names
      slide_name = x_names[idx]
      #right now choosing both randomly otherwise it can chosen carefully one sequentila and other randomly.
      s1_name = np.random.choice(np.array(x_names), p = self.prob_matrix[idx])
      s2_name = np.random.choice(np.array(x_names), p = self.prob_matrix[idx])
      while s2_name==s1_name:
        s2_name = np.random.choice(np.array(x_names), p = self.prob_matrix[idx])
      s1_idx = x_names.index(s1_name)
      s2_idx = x_names.index(s2_name)
      x_train = self.x_train
      q = x_train[idx]
      s1 = x_train[s1_idx]
      s2 = x_train[s2_idx]
      return (q,s1,s2, slide_name, s1_name, s2_name)





