import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

def write_csv(filename, array):
    with open(filename + '.csv', 'a') as f:
        for x in range(len(array)):
            for y in range(len(array)):
                f.write(str(array[x][y]) + ',')
            f.write('\n')

for num in [44, 50, 65]:
    KL_divergences = np.load('./LDA_KL_Divergence_' + str(num) + '_topics.npy')
    KL_divergences = scaler.fit_transform(KL_divergences)
    for x in range(len(KL_divergences)):
        for y in range(len(KL_divergences[x])):
            KL_divergences[x][y] = 1 - KL_divergences[x][y]

    np.save('LDA_similarities_' + str(num) + '_topics', KL_divergences)
    write_csv('LDA_similarities_' + str(num) + '_topics', KL_divergences)