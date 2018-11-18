#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import random

import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Conv1D, Conv2D, UpSampling2D, MaxPooling2D, MaxPooling1D, Reshape, LeakyReLU, Dropout
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint, Callback, TerminateOnNaN
from keras.optimizers import Adam, SGD
from sklearn.decomposition import PCA
from keras.layers.normalization import BatchNormalization

INPUT_SIZE = 4992
FEATURE_SIZE = np.int32(INPUT_SIZE/4)
counter = 0
batch_size = 1024
data_stddiv = 0
data_mean = 0
encoder = None
decoder = None
target = None

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

def Target():
    input_data = Input(shape=[FEATURE_SIZE])
    x = input_data
    x = Dense(int(FEATURE_SIZE), activation='tanh')(x)
    x = Dense(int(FEATURE_SIZE), activation='tanh')(x)
    x = Dense(int(FEATURE_SIZE), activation='tanh')(x)
    x = Dense(int(FEATURE_SIZE), activation='relu')(x)
    x = Dense(1)(x)
    return Model(input_data, x, name="target")

def get_training_data():
    df = pd.read_csv('..\\train.csv')
    raw_data = df.iloc[:, 2:].values
    target_data = df.iloc[:, 1].values
    return target_data, raw_data


def get_testing_data():
    df = pd.read_csv('..\\test.csv')
    raw_data = df.iloc[:, 1:].values
    raw_keys = df.iloc[:, 0].values
    return raw_keys, raw_data


def standard_normal_data(data):
    data_stddiv = np.std(data)
    data_mean = np.mean(data)
    data = (data - data_mean)/data_stddiv
    return data_mean, data_stddiv, data


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
    print('Found Encoder')
    encoder.load_weights('encoder.h5')
if os.path.isfile('target.h5'):
    print('Found Target Model')
    target.load_weights('target.h5')

# compile the model
my_optimizer2 =  SGD(lr=1e-3, momentum=1e-1, decay=1e-4)
target.compile(optimizer=my_optimizer2, loss='msle')

#get data
print('Prepairing data, this will take a moment with referance frame orbiting event horizon of a blackhole.')
target_data, train_data = get_training_data()
raw_keys, raw_data      = get_testing_data()
combined_data           = np.concatenate((train_data, raw_data))
test_data = None
# Normalize Data
data_mean, data_stddiv, combined_data = standard_normal_data(combined_data)
train_data = (train_data - data_mean) / data_stddiv
train_data = np.c_[train_data, np.zeros((train_data.shape[0]))]
train_data_enocder = encoder.predict(train_data)
target_data = np.log(target_data + 1)

# Use Encoder to caculate input to final transaction value producing model
raw_data = (raw_data - data_mean) / data_stddiv
raw_data = np.c_[raw_data, np.zeros((raw_data.shape[0]))]
raw_data_encoder = encoder.predict(raw_data)

prediction = target.predict(raw_data_encoder, verbose=1)

prediction = (prediction * data_stddiv) + data_mean
# Save results to CSV file
prediction_df = pd.DataFrame(
    data=prediction, index=raw_keys, columns=['target'])
prediction_df.to_csv('result.csv')
