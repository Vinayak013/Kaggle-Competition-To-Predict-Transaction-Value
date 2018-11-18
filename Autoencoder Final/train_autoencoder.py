8#!/usr/bin/env python3
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

# Use Keras Callbacks to print error and save Encoder,Decoder after every epoch
class AutoEncoderLossHistory(Callback):
    def __init__(self, save_after_epoch):
        self.epoch_count = 0
        self.epoch_save = save_after_epoch

    def on_train_begin(self, logs={}):
        self.losses = []
        self.min_loss = None

    def on_epoch_end(self, batch, logs={}):
        global encoder, decoder
        self.epoch_count = self.epoch_count + 1
        self.losses.append(logs.get('loss'))
        if self.min_loss is None:
            self.min_loss = logs.get('loss')
        elif logs.get('loss') < self.min_loss:
            self.min_loss = logs.get('loss')
            print('saving encoder and decoder.')
            encoder.save_weights('encodernew.h5')
            decoder.save_weights('decodernew.h5')
        plt.plot(self.losses, 'r')
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
	
def Decoder():
    input_data = Input(shape=[FEATURE_SIZE])
    x = input_data
    x = Dense(np.int32(FEATURE_SIZE*2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(np.int32(FEATURE_SIZE*2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(np.int32(FEATURE_SIZE*4), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(INPUT_SIZE)(x)
    return Model(input_data, x, name="decoder")

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
def generator_for_autoencoder(data):
    global batch_size
    data_size = data.shape[0]
    indices = [x for x in range(data_size)]
    i = 0
    while True:
        X = np.zeros((batch_size, INPUT_SIZE))
        if i < (data_size - batch_size - 1):
            X[:, 0:(INPUT_SIZE-1)] = data[indices[i:(i + batch_size)], :]
            i = i + batch_size
        else:
            X[:, 0:(INPUT_SIZE-1)] = data[indices[(data_size - batch_size):(data_size)], :]
            i = 0
            random.shuffle(indices)
        yield X, X

# define input to the model:
x = Input(shape=[INPUT_SIZE])
np.set_printoptions(threshold=np.nan)
# make the model:
encoder = Encoder()
encoder.summary()
decoder = Decoder()
decoder.summary()

# load saved models
print('loading model')
if os.path.isfile('encoder.h5'):
    print('encoder picked up')
    encoder.load_weights('encoder.h5')
if os.path.isfile('decoder.h5'):
    print('decoder picked up')
    decoder.load_weights('decoder.h5')

autoencoder = Model(x, decoder(encoder(x)))

# compile the model:
my_optimizer = SGD()
autoencoder.compile(optimizer=my_optimizer, loss='mse')

# display model
print(autoencoder.summary())

# Get Data
print('Prepairing data, this will take a moment with referance frame orbiting event horizon of a blackhole.')
target_data, train_data = get_training_data()
test_data               = get_testing_data()
# Use both test and train data for training Autoencoder
combined_data           = np.concatenate((train_data, test_data))
# Normalize Data
data_mean, data_stddiv, combined_data = standard_normal_data(combined_data)
losshistory = AutoEncoderLossHistory(save_after_epoch=50)
callbacks_list = [losshistory]

# Fit the model
autoencoder.fit_generator(generator=generator_for_autoencoder(combined_data),
                          steps_per_epoch=64, epochs=1000, callbacks=callbacks_list)
