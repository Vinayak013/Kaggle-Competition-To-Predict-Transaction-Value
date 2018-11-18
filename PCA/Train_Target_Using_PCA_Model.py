#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import random

import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Conv1D, Conv2D, UpSampling2D, MaxPooling2D, MaxPooling1D, Reshape, LeakyReLU
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint, Callback, TerminateOnNaN
from keras.optimizers import Adam, SGD
from sklearn.decomposition import PCA

INPUT_SIZE = 1023
FEATURE_SIZE = 5000
counter = 0
batch_size = 1024
data_stddiv = 0
data_mean = 0
encoder = None
decoder = None
target = None

plt.xlabel('epoches')
plt.ylabel('loss')
plt.title('Training history')


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
            encoder.save_weights('encoder.h5')
            decoder.save_weights('decoder.h5')
        plt.plot(self.losses, 'r')
        plt.pause(0.01)


class TargetLossHistory(Callback):
    def __init__(self, save_after_epoch):
        self.epoch_count = 0
        self.epoch_save = save_after_epoch

    def on_train_begin(self, logs={}):
        self.losses = []
        self.loss_min = None

    def on_epoch_end(self, batch, logs={}):
        global target
        if self.loss_min is None or self.loss_min > logs.get('loss'):
            print('saving target.')
            self.loss_min = logs.get('loss')
            target.save_weights('target.h5')
        self.losses.append(logs.get('loss'))
        plt.plot(self.losses, 'g')
        plt.pause(0.01)

def Target():
    input_data = Input(shape=[INPUT_SIZE])
    x = input_data
    x = Dense(int(FEATURE_SIZE), activation='tanh')(x)
    x = Dense(int(FEATURE_SIZE))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(int(FEATURE_SIZE), activation='relu')(x)
    x = Dense(1, activation='relu')(x)
    return Model(input_data, x, name="target")


def get_training_data():
    df = pd.read_csv('train.csv')
    raw_data = df.iloc[:, 2:].values
    target_data = df.iloc[:, 1].values
    return target_data, raw_data


def get_testing_data():
    df = pd.read_csv('test.csv')
    raw_data = df.iloc[:, 1:].values
    raw_keys = df.iloc[:, 0].values
    return raw_keys, raw_data


def standard_normal_data(data):
    data_stddiv = np.std(data)
    data_mean = np.mean(data)
    data = (data - data_mean)/data_stddiv
    return data_mean, data_stddiv, data


def scale_data(data):
    data_max = np.amax(data)
    data_min = np.amin(data)
    data = (data - data_min)/(data_max-data_min)
    return data_max, data_min, data

def generator_for_target(target_data, raw_data):
    global batch_size
    data_size = raw_data.shape[0]
    indices = [x for x in range(data_size)]
    i = 0
    while True:
        X = np.zeros((batch_size, INPUT_SIZE))
        Y = np.zeros([batch_size, 1])
        if i < (data_size - batch_size - 1):
            X[:, :] = raw_data[indices[i:(i + batch_size)], :]
            Y[:, 0] = target_data[indices[i:(i + batch_size)]]
            i = i + batch_size
        else:
            X[:, :] = raw_data[indices[(
                data_size - batch_size):(data_size)], :]
            Y[:, 0] = target_data[indices[(
                data_size - batch_size):(data_size)]]
            i = 0
            random.shuffle(indices)
        yield X, Y


# define input to the model:
x = Input(shape=[INPUT_SIZE])
np.set_printoptions(threshold=np.nan)
# Define the Model
target = Target()
target.summary()

# Load any model if present
print('loading model')
if os.path.isfile('target.h5'):
    target.load_weights('target.h5')

# compile the model:
my_optimizer2 =  SGD(lr=1e-3, momentum=1e-1, decay=1e-4)
target.compile(optimizer=my_optimizer2, loss='msle')

#get data
print('Prepairing data, this will take a moment with referance frame orbiting event horizon of a blackhole.')
target_data, train_data = get_training_data()

# Apply PCA
with open('objs.pkl', 'rb') as handle:
    y, data_mean, data_stddiv = pickle.load(handle)
train_data = (train_data - data_mean) / data_stddiv
train_data = y.transform(train_data)
target_data = np.log(target_data + 1)

losshistory = TargetLossHistory(save_after_epoch=10)
callbacks_list = [losshistory]
# fit model
target.fit_generator(generator=generator_for_target(target_data, train_data),
                     steps_per_epoch=64, epochs=200, callbacks=callbacks_list)

# Using trained model predict for test data
print('Ground truth:', target_data)
print('Prediction data:', target.predict(train_data))
print('loading test data.')
test_keys, test_data = get_testing_data()
test_data = (test_data - data_mean) / data_stddiv
test_data = y.transform(test_data)
prediction = target.predict(test_data)
prediction = np.exp(prediction) - 1
prediction_df = pd.DataFrame(
    data=prediction, index=test_keys, columns=['target'])
# Save results to csv file to be submitted
prediction_df.to_csv('result.csv')
