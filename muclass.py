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
from tensorflow.keras.callbacks import Callback,EarlyStopping,ReduceLROnPlateau

import keras_tuner
from keras_tuner import HyperParameters
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'



seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
#%%
def load_data():
    dataset = pd.read_csv("/data/music_data/compressed_wavs_1378_split_int.csv",header=None,index_col=None)
    X = (dataset.drop(dataset.columns[0], axis=1, inplace=False) + 32768)/65535
    Y = dataset[dataset.columns[0]]
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(Y)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,encoded_Y,test_size=0.2,random_state=seed)
    Xtrain = Xtrain.values.reshape(len(Xtrain),len(Xtrain.columns),1)
    Xtest = Xtest.values.reshape(len(Xtest),len(Xtest.columns),1)
    
    return Xtrain, Xtest, Ytrain, Ytest
#%%

class LRCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        # decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        #lr_with_decay = lr / (1. + decay * tf.keras.backend.cast(iterations, tf.keras.backend.dtype(decay)))
        print("\n Learning Rate = ", lr(iterations))
        print("Iterations = ", iterations)

print_lr = LRCallback()
#%%

class HyperStudent(keras_tuner.HyperModel):

    def __init__(self, input_shape, training_loss, param_threshold=(4500,5000)):#  4500,5000
        self.input_shape = input_shape
        self.training_loss = training_loss
        self.num_layers = [2,3,4]
        self.num_params = [4,8,16,32, 64]

        model_configurations = []
        self.model_configurations = []
        self.num_conf_perNlayer = []
        self.idx_confs_perNlayer = [0]

        for nl in self.num_layers:
            grid_choices = np.tile(self.num_params, (nl,1))
            configs = np.array(np.meshgrid(*grid_choices)).T.reshape(-1, nl)

            model_configurations.append(configs.tolist())

        model_configurations = [num for sublist in model_configurations for num in sublist]

        lastconflen = 0
        
        for config in model_configurations:
            params = self.compute_model_params(config)
            if params <= param_threshold[1] and params >= param_threshold[0]:
                self.model_configurations.append(config)
                if len(config) != lastconflen and lastconflen != 0:
                    # Counting the number of configurations for each number of layers (it is very likely that there is a better way to do it)
                    self.idx_confs_perNlayer.append(len(self.model_configurations) - 1) 
                lastconflen = len(config)
        self.idx_confs_perNlayer.append(len(self.model_configurations))
        self.num_conf_perNlayer = [j-i for i, j in zip(self.idx_confs_perNlayer[:-1], self.idx_confs_perNlayer[1:])]
    
        print('Total feasible configurations: ', len(self.model_configurations))

    def compute_model_params(self, config):
        total_params = 0
        total_params += np.prod(self.input_shape[1:])*config[0]
        total_params += config[-1]
        for i in range(len(config)-1):
            total_params += config[i]*config[i+1]
        return total_params 


    def build(self, hp):

        #quantized = True
        numframes = Xtrain.shape[1]
        
        subframes = list(factorint(numframes))[-1]
        
        inputs = tf.keras.layers.Input(shape=(numframes,1,),name="input_1")
        
        x = tf.keras.layers.LSTM(64,input_shape=[None,numframes,1],return_sequences=False)(inputs)

        config_index = hp.Int("config_indx", min_value=0, max_value=len(self.model_configurations)-1, step=1)

        # Number of hidden layers of the MLP is a hyperparameter.
        
        for units in self.model_configurations[config_index]:
            # Number of units of each layer are
            # different hyperparameters with different names.
            x = tf.keras.layers.Dense(units=units,activation='relu')(x)
        
        # The last layer contains 1 unit, which
        # represents the learned loss value
        
        
        outputs = tf.keras.layers.Dense(10,activation='softmax')(x)

        hyper_student = tf.keras.Model(inputs=inputs, outputs=outputs)

        hyper_student.compile(
            optimizer=Adam(lr=0.001, amsgrad=True),
            loss=self.training_loss
            )
        hyper_student.summary()
        return hyper_student


#%%

def optimisation(x_train,y_train, training_loss):

    hypermodel = HyperStudent(x_train.shape, training_loss)
    #tuner = keras_tuner.Hyperband(
    #      hypermodel,
    #      objective='val_loss',
    #      #max_trials=len(hypermodel.model_configurations),
    #      overwrite=True,
    #      directory='output/hyper_tuning',
    #      max_epochs=100
    #      )

    tuner = keras_tuner.RandomSearch(
          hypermodel,
          objective='val_loss',
          max_trials=len(hypermodel.model_configurations),
          overwrite=True,
          directory='output/hyper_tuning',
          )
    tuner.search_space_summary()
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1e-9)
        ]
    tuner.search(
        x=x_train,
        y=y_train,
        epochs=3,
        batch_size=1500,
        validation_split=0.2,
        callbacks=callbacks
        )

    tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    
    print('Optimal Configuration:', hypermodel.model_configurations[best_hps['config_indx']])




#%%

def final_model(Xtrain, Xtest, Ytrain, Ytest):
    numframes = Xtrain.shape[1]
    
    subframes = list(factorint(numframes))[-1]
    
    inputs = tf.keras.layers.Input(shape=(numframes,1,),name="input_1")
    
    x = tf.keras.layers.LSTM(64,input_shape=[None,numframes,1],return_sequences=False)(inputs)
    #simple_rnn = tf.keras.layers.SimpleRNN(10,input_shape=[None, 1])
    #x = tf.keras.layers.LSTM(32,return_sequences=False)(x)
    # x = tf.keras.layers.Dense(128,activation='relu')(x)
    # x = tf.keras.layers.Dropout(.1)(x)
    x = tf.keras.layers.Dense(32,activation='relu')(x)
    x = tf.keras.layers.Dense(64,activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Dense(128,activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Dense(32,activation='relu')(x)
    x = tf.keras.layers.Dense(10,activation='softmax')(x)
    #output = simple_rnn(Xtrain.values.reshape(len(Xtrain),len(Xtrain.columns),1)/255)
    
    output = x
    
    muclass = tf.keras.Model(inputs=inputs,outputs=output,name="rnn_model")
    
    muclass.summary()
    initial_lr=0.001
    learning_schedule = ExponentialDecay(initial_lr,decay_steps=1000,decay_rate=0.9,staircase=False)
    adam = Adam(learning_rate=learning_schedule)
    muclass.compile(optimizer=adam,
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    
    num_epochs = 1
    batch = 1500
    
    history = muclass.fit(Xtrain, Ytrain, epochs=num_epochs,verbose=1,validation_data=(Xtest, Ytest),batch_size=batch,callbacks=[print_lr])
    #,callbacks=[print_lr]
    return muclass,history

def plotter(history):
    fig,ax = plt.subplots()
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_title('Training and Validation loss per epoch')
    ax.set_xlabel('# Epoch')
    ax.set_ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    fig2,ax2 = plt.subplots()
    ax2.plot(history.history['accuracy'], label='acc')
    ax2.plot(history.history['val_accuracy'], label='val_acc')
    ax2.set_title('Training and Validation accuracy per epoch')
    ax2.set_xlabel('# Epoch')
    ax2.set_ylabel('accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
#%%

Xtrain, Xtest, Ytrain, Ytest = load_data()

optimisation(Xtrain,Ytrain, 'sparse_categorical_crossentropy')
#model,history = final_model(Xtrain, Xtest, Ytrain, Ytest)

#plotter(history)

