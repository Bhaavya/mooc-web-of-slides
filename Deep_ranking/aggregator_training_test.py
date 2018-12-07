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
from model import *
from preprocess_data import *

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
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


#five-fold division
fivefold_test = [[i:i+y] for i in range(0, len(X), int(len(X)/5))] 
s = range(len(X))
fivefold_train = [list(set(s)-set(x)) for x in five_fold_test]
I_permuatation = np.random.permutation(len(X))
fivefold_train = [list(map(lambda l : X[I_permuatation[l]], x)) for x in fivefold_train]
fivefold_test = [list(map(lambda l : X[I_permuatation[l]], x)) for x in fivefold_test]
x_train = fivefold_train[0]
x_test = fivefold_test[0] 
x_val = x_test[:int(len(x_test)/2)]
x_test = x_test[int(len(x_test)/2):]
fivefold_train_names = [list(map(lambda l : X_names[I_permuatation[l]], x)) for x in fivefold_train]
fivefold_test_names = [list(map(lambda l : X_names[I_permuatation[l]], x)) for x in fivefold_test]
x_train_names = fivefold_train_names[0]
x_test_names = fivefold_test_names[0]
x_val_names = x_val_names[:int(len(x_test_names)/2)]
v_test_names = x_val_names[int(len(x_test_names)/2):]
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
num_epochs = 20 if args.max_epoch is None else args.max_epoch
optimi = 'SGD' if args.SGD else 'ADAM'
lr = lr_dict.get(optim,0.001) if args.lr is None else args.lr
#embedding_size = 500 if args.embed_size is None else args.embed_size
num_hidden_units = 500 if args.num_hidden_units is None else args.num_hidden_units
savefile = '_'.join(['Aggregator_model', str(batch_size), str(num_epochs), optimi])


heuristic_features = Heuristic_features()

X_heuristic_features = heuristic_features.get_features(x_train, x_train_names)


model = aggregator_model(num_features, num_hidden_units)
model = model.to(device)
model.train()

# Dataset and loader
train_dataset = triplettrainDataset_aggegator(x_train, x_train_names)


if(optimi=='ADAM'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif(optimi=='SGD'):
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum =0.9)

total_step = len()
training_loss = []
zero = torch.zeros(batch_size).to(device)

for epoch in range(num_epochs):
  running_loss = 0.0
  epoch_counter = 0
    
  for i, (s,s1,s2) in enumerate(train_loader):
    q_image = q_image.to(device)
    p_image = p_image.to(device)
    n_image = n_image.to(device)
    q_output = model(q_image)
    p_output = model(p_image)
    n_output = model(n_image)
    loss = criterion(zero, q_output, p_output, n_output)
    #loss2 = criterion2(q_output, p_output, n_output)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if (i+1) % 100 == 0:
      print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, running_loss/(counter/batch_size) ))
    epoch_counter += batch_size
  training_loss.append((running_loss)/(epoch_counter/batch_size))
  print ('Epoch , Loss: {:.4f}'.format(epoch+1, running_loss/(counter/batch_size)))
  torch.save(model, 'model/'+savefile+'_temp.model')
  np.save('state/' + savefile + 'training_loss.npy', np.array(training_loss))
  #for param_group in optimizer.param_groups:
  #  param_group['lr'] = 0.001/(1+((15+epoch)*0.05))
  #  print (param_group['lr'])
      
  #training_epoch_loss += [loss.item()]
torch.save(model, 'model/'+savefile+'.model')
np.save('state/' + savefile + 'training_loss.npy', np.array(training_loss))

