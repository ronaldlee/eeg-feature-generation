#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional,BatchNormalization,PReLU,ReLU,Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model


init_df = pd.read_csv('csv/out_gameemo.csv',  sep=',')

print('Shape of data: ', init_df.shape)


# In[2]:




#HA_PV = high arousal, positive valence
#HA_NV = high arousal, negative valence
#LA_NV = low arousal, negative valence
#LA_PV = low arousal, positive valance
label_map = {1:"HA_PV", 2:"HA_NV", 3:"LA_NV", 4:"LA_PV"}

init_df["Label"] = init_df["Label"].map(label_map)

print(init_df.head())

features = init_df.iloc[:, :-1]
label = init_df.iloc[:, -1:]

print('Shape of data: ', init_df.shape)
print('features.shape: ', features.shape)
print('label.shape: ', label.shape)

init_df.head()
print(init_df.columns)


y = label
X = features

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=48)

X_train = np.array(X_train).reshape((X_train.shape[0],X_train.shape[1],1))
X_test = np.array(X_test).reshape((X_test.shape[0],X_test.shape[1],1))

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


# In[3]:


def train_model(x_train, y_train,x_test,y_test, save_to, epoch = 2):
    strategy = tf.distribute.MirroredStrategy(devices=None)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        inputs = tf.keras.Input(shape=(X_train.shape[1],1))

        #ml_model = tf.keras.layers.GRU(256, return_sequences=True)(inputs)
        ml_model = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)

        flat = Flatten()(ml_model)
        outputs = Dense(4, activation='softmax')(flat)
        model = tf.keras.Model(inputs, outputs)

        #model = tf.keras.models.load_model('_best_model.h5')

        model.summary()
        tf.keras.utils.plot_model(model)

        opt_adam = keras.optimizers.Adam(learning_rate=0.001)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(save_to + '_best_model_lstm_all_cat.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
            
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * np.exp(-epoch / 10.))
            
        model.compile(optimizer=opt_adam,
                      loss=['categorical_crossentropy'],
                      metrics=['accuracy'])
        
    history = model.fit(x_train,y_train,
                        batch_size=32,
                        epochs=epoch,
                        validation_data=(x_test,y_test),
                        callbacks=[es,mc,lr_schedule])
        
    saved_model = load_model(save_to + '_best_model_lstm_all_cat.h5')
        
    return model,history


# In[4]:



model,history = train_model(X_train, y_train,X_test, y_test, save_to= './', epoch = 40)


# In[ ]:




