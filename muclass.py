#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sympy.ntheory import factorint
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

dataset = pd.read_csv("/data/music_data/compressed_wavs_1378_split_int.csv",header=None,index_col=None)

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
#%%

#dataset = dataset.sample(frac=1).reset_index(drop=True)
X = (dataset.drop(dataset.columns[0], axis=1, inplace=False) + 32768)/65535
Y = dataset[dataset.columns[0]]
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,encoded_Y,test_size=0.2,random_state=seed)
Xtrain = Xtrain.values.reshape(len(Xtrain),len(Xtrain.columns),1)
Xtest = Xtest.values.reshape(len(Xtest),len(Xtest.columns),1)
#%%
numframes = Xtrain.shape[1]

subframes = list(factorint(numframes))[-1]

inputs = tf.keras.layers.Input(shape=(numframes,1,),name="input_1")

x = tf.keras.layers.LSTM(64,input_shape=[None,numframes,1],return_sequences=False)(inputs)
#simple_rnn = tf.keras.layers.SimpleRNN(10,input_shape=[None, 1])
x = tf.keras.layers.Dense(32,activation='relu')(x)
x = tf.keras.layers.Dense(10,activation='softmax')(x)
#output = simple_rnn(Xtrain.values.reshape(len(Xtrain),len(Xtrain.columns),1)/255)

output = x

muclass = tf.keras.Model(inputs=inputs,outputs=output,name="rnn_model")

muclass.summary()
initial_lr=0.001
learning_schedule = ExponentialDecay(initial_lr,decay_steps=100000,decay_rate=0.99,staircase=False)
adam = Adam(learning_rate=learning_schedule)
muclass.compile(optimizer=adam,
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

num_epochs = 100
batch = 1024

history = muclass.fit(Xtrain, Ytrain, epochs=num_epochs,verbose=1,validation_data=(Xtest, Ytest),batch_size=batch)

fig,ax = plt.subplots()
ax.plot(history.history['loss'], label='loss')
ax.plot(history.history['val_loss'], label='val_loss')
ax.set_title('Training and Validation loss per epoch')
ax.set_xlabel('# Epoch')
ax.set_ylabel('loss')
plt.legend()
plt.tight_layout()
plt.show()