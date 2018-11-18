#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import random

import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Conv1D, Conv2D, UpSampling2D, MaxPooling2D, MaxPooling1D, Reshape
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization

# Important Network Variables
INPUT_SIZE   = 4992 # 
FEATURE_SIZE = np.int32(INPUT_SIZE/4) # Encoder o/p size
batch_size   = 1024 # Batch size used in tarining

counter     = 0
data_stddiv = 0
data_mean   = 0
encoder     = None
decoder     = None
target      = None

plt.xlabel('epoches')
plt.ylabel('loss')
plt.title('Training history')

# Use Keras Callbacks to print error and save final model predicting transaction value after every epoch
class TargetLossHistory(Callback):
    def __init__(self, save_after_epoch):
        self.epoch_count = 0
        self.epoch_save = save_after_epoch

    def on_train_begin(self, logs={}):
        self.losses = []
        self.min_loss = None

    def on_epoch_end(self, batch, logs={}):
        global target
        self.epoch_count = self.epoch_count + 1
        if self.min_loss is None:
            self.min_loss = logs.get('loss')
        elif logs.get('loss') < self.min_loss:
            self.min_loss = logs.get('loss')
            print('saving target.')
            target.save_weights('target.h5')
        self.losses.append(logs.get('loss'))
        plt.plot(self.losses, 'g')
        plt.pause(0.01)

def Encoder():
    input_data = Input(shape=[INPUT_SIZE])
    x = input_data
    x = Dense(np.int32(INPUT_SIZE*2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(np.int32(INPUT_SIZE), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(np.int32(INPUT_SIZE/2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(np.int32(INPUT_SIZE/2), activation='relu')(x)	
    x = BatchNormalization()(x)
    x = Dense(np.int32(INPUT_SIZE/4), activation='relu')(x)
    x = Dense(FEATURE_SIZE)(x)
    return Model(input_data, x, name="encoder")

# Final Transaction value predicting model. Output of Encoder is input to this moddel
def Target():
    input_data = Input(shape=[FEATURE_SIZE])
    x = input_data
    x = Dense(int(FEATURE_SIZE), activation='tanh')(x)
    x = Dense(int(FEATURE_SIZE), activation='tanh')(x)
    x = Dense(int(FEATURE_SIZE), activation='tanh')(x)
    x = Dense(int(FEATURE_SIZE), activation='tanh')(x)
    x = Dense(1)(x)
    return Model(input_data, x, name="target")

# Get Data
def get_testing_data():
    df = pd.read_csv('..\\test.csv')
    raw_data = df.iloc[:, 1:].values
    return raw_data	

def get_training_data():
    df = pd.read_csv('..\\train.csv')
    raw_data = df.iloc[:, 2:].values
    target_data = df.iloc[:, 1].values
    return target_data, raw_data

def standard_normal_data(data):
    data_stddiv = np.std(data)
    data_mean = np.mean(data)
    data = (data - data_mean)/data_stddiv
    return data_mean, data_stddiv, data


# Generatore outputs single batch per every iteration
def generator_for_target(target_data, raw_data):
    global batch_size
    data_size = raw_data.shape[0]
    indices = [x for x in range(data_size)]
    i = 0
    while True:
        X = np.zeros((batch_size, FEATURE_SIZE))
        Y = np.zeros([batch_size, 1])
        if i < (data_size - batch_size - 1):
            X[:, :] = raw_data[indices[i:(i + batch_size)], :]
            Y[:, 0] = target_data[indices[i:(i + batch_size)]]
            i = i + batch_size
        else:
            X[:, :] = raw_data[indices[(data_size - batch_size):(data_size)], :]
            Y[:, 0] = target_data[indices[(data_size - batch_size):(data_size)]]
            i = 0
            random.shuffle(indices)
        yield X, Y

# define input to the model:
x = Input(shape=[INPUT_SIZE])
np.set_printoptions(threshold=np.nan)
# make the model:
encoder = Encoder()
encoder.summary()
target = Target()
target.summary()

# load saved models
print('loading model')
if os.path.isfile('encoder.h5'):
    print('encoder picked up')
    encoder.load_weights('encoder.h5')
if os.path.isfile('target.h5'):
    target.load_weights('target.h5')

# compile the model:
my_optimizer = SGD()
target.compile(optimizer=my_optimizer, loss='msle')

# Get Data
print('Prepairing data, this will take a moment with referance frame orbiting event horizon of a blackhole.')
target_data, train_data = get_training_data()
test_data               = get_testing_data()
# Use both test and train data for training Autoencoder
combined_data           = np.concatenate((train_data, test_data))
data_mean, data_stddiv, combined_data = standard_normal_data(combined_data)


test_data = None

losshistory = TargetLossHistory(save_after_epoch=50)
callbacks_list = [losshistory]
train_data = (train_data - data_mean)/data_stddiv
# use encoder to generate data
data = np.zeros((train_data.shape[0], INPUT_SIZE))
data[:, 0:(INPUT_SIZE - 1)] = train_data
#target_data = np.log(target_data + 1)
target_data = (target_data - data_mean)/data_stddiv
train_data = encoder.predict(data)

# fit model
target.fit_generator(generator=generator_for_target(target_data, train_data),
                     steps_per_epoch=128, epochs=1000, callbacks=callbacks_list)
plt.show()
