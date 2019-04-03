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
import sys
sys.path.insert(0, '/Users/sahiti/Documents/gitlab_repos/mooc-web-of-slides/evaluation/')
import seq_eval

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print (device)

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
X_names = []
with io.open('preprocessed_data/X_names.txt','r',encoding='utf-8') as f:
  lines = f.readlines()
for line in lines:
    line = line.strip()
    X_names.append(line)


#five-fold division
fivefold_test = [range(i,(i+int(len(X)/5))) for i in range(0, len(X), int(len(X)/5))] 
s = range(len(X))
fivefold_train = [list(set(s)-set(x)) for x in fivefold_test]
I_permuatation = np.random.permutation(range(len(X)))
fivefold_train_ = [[X[I_permuatation[l]] for l in fivefold_train[0]]]
fivefold_test_ = [[X[I_permuatation[l]] for l in fivefold_test[0]]]
x_train = fivefold_train[0]
x_test = fivefold_test[0] 
x_val = x_test[:int(len(x_test)/2)]
x_test = x_test[int(len(x_test)/2):]
fivefold_train_names = [[X_names[I_permuatation[l]] for l in fivefold_train[0]]]
fivefold_test_names = [[X_names[I_permuatation[l]] for l in  fivefold_test[0]]]
x_train_names = fivefold_train_names[0]
x_test_names = fivefold_test_names[0]
x_val_names = x_test_names[:int(len(x_test_names)/2)]
x_test_names = x_test_names[int(len(x_test_names)/2):]
## Parser
parser = argparse.ArgumentParser()
parser.add_argument('-n','--batch_size', type=int)
parser.add_argument('-t','--max_epoch', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--SGD', action='store_true')
#parser.add_argument('--embed_size', type=int)
parser.add_argument('--num_hidden_units', type=int)
args = parser.parse_args()

lr_dict = {'SGD': 0.001, 'ADAM':0.01}
batch_size = 100 if args.batch_size is None else args.batch_size
num_epochs = 5 if args.max_epoch is None else args.max_epoch
optimi = 'SGD' if args.SGD else 'ADAM'
lr = 0.5 if args.lr is None else args.lr
#embedding_size = 500 if args.embed_size is None else args.embed_size
num_hidden_units = 500 if args.num_hidden_units is None else args.num_hidden_units
print lr

heuristic_features = Heuristic_features()
num_features = heuristic_features.__num_features__()
print num_features

savefile = '_'.join(['Aggregator_model', str(batch_size), str(num_epochs), str(num_features),str(num_hidden_units)])

model = aggregator_model(num_features, num_hidden_units)
model = model.to(device)
model.train()

# Dataset and loader
train_dataset = triplettrainDataset_aggregator(x_train, x_train_names)


criterion = nn.MarginRankingLoss(margin =1.0)
criterion = criterion.to(device)
if(optimi=='ADAM'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif(optimi=='SGD'):
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum =0.9)

L_train = len(x_train_names)
total_step = int(L_train/batch_size)
training_loss = []

for epoch in range(num_epochs):
  running_loss = 0.0
  epoch_counter = 0
  
  I_permutation = np.random.permutation(L_train)
  
  for i in range(0, L_train, batch_size):
    print (epoch_counter)
    x_input2 = [train_dataset.__getitem__(j) for j in I_permutation[i:i+batch_size]]
    bs = len(x_input2)
    q = np.zeros((bs,num_features),dtype=np.float)
    p = np.zeros((bs,num_features),dtype=np.float)
    t = np.zeros((bs),dtype = np.float)
    for j in range(bs):
      q[j,:] = heuristic_features.get_features(x_input2[j][3],x_input2[j][4])
      p[j,:] = heuristic_features.get_features(x_input2[j][3],x_input2[j][5])
      y_q = int(seq_eval.is_seq(x_input2[j][3],x_input2[j][4]) == True) 
      y_p = int(seq_eval.is_seq(x_input2[j][3],x_input2[j][5]) == True) 
      if(y_q > y_p):
        target = 1
      elif(y_q<y_p):
        target = -1
      else: 
        target = 0 
      t[j] = target
    #print (t)
    t = torch.FloatTensor(t).to(device) 
    q = torch.FloatTensor(q).to(device)
    p = torch.FloatTensor(p).to(device)
    q_output = model(q)
    p_output = model(p)
    loss = criterion(q_output,p_output, t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if (i+1) % 100 == 0:
      print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, running_loss/(epoch_counter/batch_size) ))
    epoch_counter += bs
  training_loss.append((running_loss)/(epoch_counter/batch_size))
  print ('Epoch {}, Loss: {:.4f}'.format(epoch+1, running_loss/(epoch_counter/batch_size)))
  torch.save(model, 'model/'+savefile+'_temp.model')
  np.save('state/' + savefile + 'training_loss.npy', np.array(training_loss))

torch.save(model, 'model/'+savefile+'.model')
np.save('state/' + savefile + 'training_loss.npy', np.array(training_loss))

'''
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
  print k    

np.save("../data/"+savefile+"_sim.npy",sim_mat)
'''