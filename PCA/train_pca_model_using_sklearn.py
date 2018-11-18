from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import pickle

def get_training_data():
    df = pd.read_csv('train.csv')
    raw_data = df.iloc[:, 2:].values
    target_data = df.iloc[:, 1].values
    return target_data, raw_data

def get_testing_data():
    df = pd.read_csv('test.csv')
    raw_data = df.iloc[:, 1:].values
    return raw_data

def standard_normal_data(data):
    data_stddiv = np.std(data)
    data_mean = np.mean(data)
    data = (data - data_mean)/data_stddiv
    return data_mean, data_stddiv, data

# Get Data
print('Prepairing data, this will take a moment with referance frame orbiting event horizon of a blackhole.')
target_data, train_data = get_training_data()
test_data = get_testing_data()
combined_data = np.concatenate((train_data, test_data))
test_data = None
data_mean, data_stddiv, combined_data = standard_normal_data(combined_data)
target_data = (combined_data - data_mean)/data_stddiv

print('Now PCA')
pca = PCA(n_components=np.int32(4092/4))
y = pca.fit(target_data)
with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([y, data_mean, data_stddiv], f)