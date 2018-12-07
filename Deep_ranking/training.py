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

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print (device)

slide_dictionary = np.load('preprocessed_data/slide_dictionary.npy')
vocab_size = len(slide_dictionary)
vocab_size += 1
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

#five-fold division
fivefold_test = [range(i,(i+int(len(X)/5))) for i in range(0, len(X), int(len(X)/5))] 
s = range(len(X))
fivefold_train = [list(set(s)-set(x)) for x in fivefold_test]
I_permuatation = np.random.permutation(range(len(X)))
#print (fivefold_train[0])
#print (fivefold_test[0])
#print (I_permuatation[904:])
fivefold_train_ = [[X[I_permuatation[l]] for l in fivefold_train[0]]]
fivefold_test_ = [[X[I_permuatation[l]] for l in fivefold_test[0]]]
x_train = fivefold_train_[0]
x_test = fivefold_test_[0] 
fivefold_train_names = [[X_names[I_permuatation[l]] for l in fivefold_train[0]]]
fivefold_test_names = [[X_names[I_permuatation[l]] for l in  fivefold_test[0]]]
x_train_names = fivefold_train_names[0]
x_test_names = fivefold_test_names[0]

#FOR NOE TRAIN ON ENTIRE DATASET LIKE AN UNSUPERVISED MODEL...
x_train = x_train+x_test
x_train_names = x_train_names + x_test_names
## Parser
parser = argparse.ArgumentParser()
parser.add_argument('-n','--batch_size', type=int)
parser.add_argument('-t','--max_epoch', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--SGD', action='store_true')
#parser.add_argument('--embed_size', type=int)
parser.add_argument('--num_hidden_units1', type=int)
parser.add_argument('--num_hidden_units2', type=int)
args = parser.parse_args()

lr_dict = {'SGD': 0.001, 'ADAM':0.01}
batch_size = 100 if args.batch_size is None else args.batch_size
num_epochs = 10 if args.max_epoch is None else args.max_epoch
optimi = 'SGD' if args.SGD else 'ADAM'
lr = lr_dict.get(optim,0.001) if args.lr is None else args.lr
#embedding_size = 500 if args.embed_size is None else args.embed_size
num_hidden_units1 = 500 if args.num_hidden_units1 is None else args.num_hidden_units1
num_hidden_units2 = 500 if args.num_hidden_units2 is None else args.num_hidden_units2
alphas = [0.5, 0.5] #weights of different similarities
savefile = '_'.join(['RNN_model', str(batch_size), str(num_epochs), optimi])



model = RNN_model(vocab_size, num_hidden_units1)
model = model.to(device)
model.train()

# Dataset and loader
train_dataset = triplettrainDataset(x_train, x_train_names, alphas)
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                           batch_size=batch_size, 
#                                           shuffle=True)


criterion = Similarityloss()
criterion = criterion.to(device)
criterion2 = nn.TripletMarginLoss(margin = 1.0, p=2)
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
    sequence_length = 75
    bs = len(x_input2)
    zero = torch.zeros(bs).to(device)
    x_input = np.zeros((bs,sequence_length),dtype=np.int)
    for j in range(bs):
        x = np.asarray(x_input2[j][0])
        sl = x.shape[0]
        if(sl < sequence_length):
            x_input[j,0:sl] = x
        else:
            start_index = np.random.randint(sl-sequence_length+1)
            x_input[j,:] = x[start_index:(start_index+sequence_length)]
    q = x_input
    x_input = np.zeros((bs,sequence_length),dtype=np.int)
    for j in range(bs):
        x = np.asarray(x_input2[j][1])
        sl = x.shape[0]
        if(sl < sequence_length):
            x_input[j,0:sl] = x
        else:
            start_index = np.random.randint(sl-sequence_length+1)
            x_input[j,:] = x[start_index:(start_index+sequence_length)]
    p = x_input
    x_input = np.zeros((bs,sequence_length),dtype=np.int)
    for j in range(bs):
        x = np.asarray(x_input2[j][2])
        sl = x.shape[0]
        if(sl < sequence_length):
            x_input[j,0:sl] = x
        else:
            start_index = np.random.randint(sl-sequence_length+1)
            x_input[j,:] = x[start_index:(start_index+sequence_length)]
    n = x_input
    q = torch.LongTensor(q).to(device)
    p = torch.LongTensor(p).to(device)
    n = torch.LongTensor(n).to(device)
    q_output = model(q)
    p_output = model(p)
    n_output = model(n)
    loss = criterion(zero, q_output, p_output, n_output)
    #loss2 = criterion2(q_output, p_output, n_output)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if (i+1) % 100 == 0:
      print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, running_loss/(epoch_counter/batch_size) ))
    epoch_counter += bs
  training_loss.append((running_loss)/(epoch_counter/batch_size))
  print ('Epoch , Loss: {:.4f}'.format(epoch+1, running_loss/(epoch_counter/batch_size)))
  torch.save(model, 'model/'+savefile+'_temp.model')
  np.save('state/' + savefile + 'training_loss.npy', np.array(training_loss))
  #for param_group in optimizer.param_groups:
  #  param_group['lr'] = 0.001/(1+((15+epoch)*0.05))
  #  print (param_group['lr'])
      
  #training_epoch_loss += [loss.item()]
torch.save(model, 'model/'+savefile+'.model')
np.save('state/' + savefile + 'training_loss.npy', np.array(training_loss))

