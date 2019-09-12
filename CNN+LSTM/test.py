import csv
import pandas  as pd
import numpy as np
import sys
import torch
import random
import os
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import plot_model

#file_a='data/XJTLU/Dataset_4floor_MIX2.csv'                 #Is_file_a=input_mode.find('a')                                           
#file_b='data/XJTLU/Dataset_4floor_HuaWei.csv'

#data_a=pd.read_csv(file_a)
#data_a=data_a.iloc[:,1:525].drop(columns=['Floor','Building','Model'])
#print(data_a.columns)

from keras.models import Model
from keras.layers import Input ,Reshape,concatenate
from keras.layers import LSTM,Convolution2D,ZeroPadding2D,LeakyReLU	,MaxPooling2D,Dropout,GlobalMaxPooling2D ,Dense
from numpy import array
from keras.models import Sequential
from keras.layers import Concatenate


tweet_a = Input(shape=(280, 256))
tweet_b = Input(shape=(280, 256))


# This layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64)

# When we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# We can then concatenate the two vectors:
merged_vector = concatenate([encoded_a, encoded_b], axis=-1)

# And add a logistic regression on top
predictions = Dense(1, activation='sigmoid')(merged_vector)

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])



plot_model(model, to_file='model.png')