import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class StatefulLSTM(nn.Module):
    def __init__(self,in_size,out_size):
        super(StatefulLSTM,self).__init__()
        
        self.lstm = nn.LSTMCell(in_size,out_size)
        self.out_size = out_size
        
        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self,x):

        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.c = Variable(torch.zeros(state_size)).to(device)
            self.h = Variable(torch.zeros(state_size)).to(device)
        self.h, self.c = self.lstm(x,(self.h,self.c))

        return self.h

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):  # dropout = 0.8 is also tried
        if train==False:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)

        return mask * x

class RNN_model(nn.Module):
    def __init__(self,vocab_size, no_of_hidden_units, embedding_size = 100):
        super(RNN_model, self).__init__()

        
        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)#,padding_idx=0)

        self.lstm1 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        self.bn_lstm1= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout() #torch.nn.Dropout(p=0.5)

        #self.lstm2 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        #self.bn_lstm2= nn.BatchNorm1d(no_of_hidden_units)
        #self.dropout2 = LockedDropout() #torch.nn.Dropout(p=0.5)

        #self.lstm3 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        #self.bn_lstm3= nn.BatchNorm1d(no_of_hidden_units)
        #self.dropout3 = LockedDropout() #torch.nn.Dropout(p=0.5)

        self.decoder = nn.Linear(no_of_hidden_units,embedding_size)
        
    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        #self.lstm2.reset_state()
        #self.dropout2.reset_state()
        #self.lstm3.reset_state()
        #self.dropout3.reset_state()

    def forward(self, x, train=True):


        embed = self.embedding(x) # batch_size, time_steps, features

        no_of_timesteps = embed.shape[1]

        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps):

            h = self.lstm1(embed[:,i,:])
            h = self.bn_lstm1(h)
            h = self.dropout1(h,dropout=0.3,train=train)

        #    h = self.lstm2(h)
        #    h = self.bn_lstm2(h)
        #    h = self.dropout2(h,dropout=0.3,train=train)

        #    h = self.lstm3(h)
        #    h = self.bn_lstm3(h)
        #    h = self.dropout3(h,dropout=0.3,train=train)

            h = self.decoder(h)

            outputs.append(h)

        outputs = torch.stack(outputs) # (time_steps,batch_size,vocab_size)
        outputs = outputs.permute(1,2,0) # (batch_size,features,time_steps)
        pool = nn.MaxPool1d(no_of_timesteps)
        outputs = pool(outputs)   #batch_size,features,1
        outputs = outputs.view(outputs.size(0),-1) #batch_size,embeddings_size(features)
        return outputs
        
class RNN_model_embeddings(nn.Module):
    def __init__(self,vocab_size, no_of_hidden_units, embedding_size = 100):
        super(RNN_model_embeddings, self).__init__()

        
        #self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)#,padding_idx=0)

        self.lstm1 = StatefulLSTM(300,no_of_hidden_units)
        self.bn_lstm1= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout() #torch.nn.Dropout(p=0.5)

        #self.lstm2 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        #self.bn_lstm2= nn.BatchNorm1d(no_of_hidden_units)
        #self.dropout2 = LockedDropout() #torch.nn.Dropout(p=0.5)

        #self.lstm3 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        #self.bn_lstm3= nn.BatchNorm1d(no_of_hidden_units)
        #self.dropout3 = LockedDropout() #torch.nn.Dropout(p=0.5)

        self.decoder = nn.Linear(no_of_hidden_units,embedding_size)
        
    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        #self.lstm2.reset_state()
        #self.dropout2.reset_state()
        #self.lstm3.reset_state()
        #self.dropout3.reset_state()

    def forward(self, x, train=True):

        no_of_timesteps = x.shape[1]

        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps):

            h = self.lstm1(x[:,i,:])
            h = self.bn_lstm1(h)
            h = self.dropout1(h,dropout=0.5,train=train)

        #    h = self.lstm2(h)
        #    h = self.bn_lstm2(h)
        #    h = self.dropout2(h,dropout=0.3,train=train)

        #    h = self.lstm3(h)
        #    h = self.bn_lstm3(h)
        #    h = self.dropout3(h,dropout=0.3,train=train)

            h = self.decoder(h)

            outputs.append(h)

        outputs = torch.stack(outputs) # (time_steps,batch_size,vocab_size)
        outputs = outputs.permute(1,2,0) # (batch_size,features,time_steps)
        pool = nn.MaxPool1d(no_of_timesteps)
        outputs = pool(outputs)   #batch_size,features,1
        outputs = outputs.view(outputs.size(0),-1) #batch_size,embeddings_size(features)
        return outputs

class BOW_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units,embedding_size):
        super(BOW_model, self).__init__()
        ## will need to define model architecture
        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)

        self.fc_hidden1 = nn.Sequential(nn.Linear(no_of_hidden_units,no_of_hidden_units),
                           nn.BatchNorm1d(no_of_hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p=0.5)
                        )
        self.fc_hidden2 = nn.Sequential(nn.Linear(no_of_hidden_units,no_of_hidden_units),
                   nn.BatchNorm1d(no_of_hidden_units),
                   nn.ReLU(),
                   nn.Dropout(p=0.5)
                )
        #self.fc_hidden2 = nn.Linear(no_of_hidden_units,no_of_hidden_units)  # extra layer tried
        #self.bn_hidden2 = nn.BatchNorm1d(no_of_hidden_units)
        #self.dropout2 = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, embedding_size)
        
    def forward(self, x):
        # will need to define forward function for when model gets called
        bow_embedding = []
        for i in range(len(x)):  #batch_size
            lookup_tensor = Variable(torch.LongTensor(x[i])).to(device)
            embed = self.embedding(lookup_tensor)
            embed = embed.mean(dim=0)       #--------DOUBTFUL---------- SHOULD SEE AGAIN IN THE TUTORIAL
            bow_embedding.append(embed)    
        bow_embedding = torch.stack(bow_embedding)
    
        h = self.fc_hidden1(bow_embedding)
        #h = self.fc_hidden2(h)
        
        h = self.fc_output(h) #batch_size,embedding_size
    
        return h

class BOW_model_embeddings(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units,embedding_size):
        super(BOW_model_embeddings, self).__init__()
        ## will need to define model architecture
        #self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)

        self.fc_hidden1 = nn.Sequential(nn.Linear(300,no_of_hidden_units),
                           nn.BatchNorm1d(no_of_hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p=0.5)
                        )
        self.fc_hidden2 = nn.Sequential(nn.Linear(no_of_hidden_units,no_of_hidden_units),
                   nn.BatchNorm1d(no_of_hidden_units),
                   nn.ReLU(),
                   nn.Dropout(p=0.5)
                )
        #self.fc_hidden2 = nn.Linear(no_of_hidden_units,no_of_hidden_units)  # extra layer tried
        #self.bn_hidden2 = nn.BatchNorm1d(no_of_hidden_units)
        #self.dropout2 = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, embedding_size)
        
    def forward(self, x):
        # will need to define forward function for when model gets called
        
        h = self.fc_hidden1(x)
        #h = self.fc_hidden2(h)
        
        h = self.fc_output(h) #batch_size,embedding_size
    
        return h


class aggregator_model(nn.Module):
    def __init__(self, in_features, no_of_hidden_units,num_layers=1):
        super(aggregator_model, self).__init__()
        layers = []
        layers.append(nn.Sequential(nn.Linear(in_features, no_of_hidden_units),
                    nn.BatchNorm1d(no_of_hidden_units),
                    nn.ReLU(),
                    nn.Dropout(p=0.5)
                    ))
        self.layers = nn.Sequential(*layers)

        self.final_layer = nn.Linear(no_of_hidden_units, 1)

    
    def forward(self, x):
        out = self.final_layer(self.layers(x))
#       out = self.linear(out)
        return out


class combined_model(nn.Module):
    def __init__(self, embed_model,aggregator):
        self.embed_model = embed_model
        self.aggregator = aggregator

    def forward(self, q,p, train = True):
        q_out = self.embed_model(q,train)
        p_out = self.embed_model(p,train)
        return self.aggregator(torch.cat((q_out,p_out),dim=1))





