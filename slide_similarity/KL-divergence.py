import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n','--file_path', type=str)
args = parser.parse_args()
file_name = -1 if args.file_path is None else args.file_path
if file_name==-1:
	print ("Please provide file_path")
save_file = file_name.split('model_')[1].split('/k')[0]
#save_file = 'LFLDA_35'
print (save_file)
slide_names = open('tmp/slide_names3.txt','r')
data_distribution = open(file_name , 'r')

topic_d  = []
lines = data_distribution
for line in lines:
	line = line.strip().split(' ')
	topic_d += [[float(l) for l in line]]
print len(topic_d)

kl_d = np.zeros((len(topic_d), len(topic_d)), dtype=np.float) 
for i in range(len(topic_d)):
	for j in range(len(topic_d)):
		kl_d[i][j] = scipy.stats.entropy(topic_d[i], topic_d[j], base=2)
	if(i%100 == 0):
		print (i)
kl_d = np.nan_to_num(kl_d)
print (np.amin(kl_d))
print (np.amax(kl_d))

kl_d[kl_d==0] = 0.00000001
kl_sim = float(1)/kl_d
scaler = MinMaxScaler(feature_range=(0.000001,1))
(x,y) = kl_sim.shape
kl_sim = scaler.fit_transform(np.reshape(kl_sim, (-1,1)))
kl_sim = np.reshape(kl_sim, (x,y))

np.save("../data/KL_divergence_"+save_file+"_sim.npy",kl_sim)



