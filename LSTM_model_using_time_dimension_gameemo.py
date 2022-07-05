#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, GRU, Bidirectional,BatchNormalization,PReLU,ReLU,Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical


# In[2]:



init_df = pd.read_csv('./csv/out_gameemo_time_domain_simple.csv',  sep=',')

print('Shape of data: ', init_df.shape)


# In[3]:



df = init_df.copy()
print(df.head())

#HA_PV = high arousal, positive valence
#HA_NV = high arousal, negative valence
#LA_NV = low arousal, negative valence
#LA_PV = low arousal, positive valance
label_map = {1:"HA_PV", 2:"HA_NV", 3:"LA_NV", 4:"LA_PV"}

df["Label"] = df["Label"].map(label_map)


# In[12]:



print(df.head())

features = df.iloc[:, :-1]
label = df.iloc[:, -1:]

print('Shape of data: ', df.shape)
print('features.shape: ', features.shape)
print('label.shape: ', label.shape)

df.head()
print(df.columns)

y = label
X = features

# 38252 is the max sample size, data collected for one participant. Can choose smaller sample size that can
# divide 38252.
# 38252 can be divided by 73 or 131, 524
sample_size = int(38252/73)  
num_of_features = 14

train_dataset_percentage = 0.7

print("sample_size:",sample_size)
print("num_of_features:",num_of_features)

total_samples_count = int(X.shape[0]/sample_size)

print("total_samples_count:", total_samples_count)


train_sample_count = int(total_samples_count * train_dataset_percentage)
test_sample_count = total_samples_count - train_sample_count

train_size = train_sample_count * sample_size
test_size = test_sample_count * sample_size

print("train size:", train_size)
print("test size:", test_size)

X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]


X_train = np.array(X_train).reshape((train_sample_count,sample_size,num_of_features))
X_test = np.array(X_test).reshape((test_sample_count,sample_size,num_of_features))

print("X_train.shape after reshape:",X_train.shape)
print("X_test.shape after reshape:",X_test.shape)

#collapse y_train and y_test to the same X sample counts instead

y_train_collapsed = np.array([])
for i in range(len(y_train)):
    if (i % sample_size == 0):
        y_train_collapsed = np.append(y_train_collapsed, (y_train.iloc[i]))
        
print("y_train_collapsed shape:",y_train_collapsed.shape)        

y_test_collapsed = np.array([])
for i in range(len(y_test)):
    if (i % sample_size == 0):
        y_test_collapsed = np.append(y_test_collapsed, (y_test.iloc[i]))
        
print("y_test_collapsed shape:",y_test_collapsed.shape)    


y_train = pd.get_dummies(y_train_collapsed)
y_test = pd.get_dummies(y_test_collapsed)

print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)


# In[13]:


def train_model(x_train, y_train,x_test,y_test, save_to, epoch, sample_size, num_of_features):
    strategy = tf.distribute.MirroredStrategy(devices=None)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    with strategy.scope():

#         inputs = tf.keras.Input(shape=(sample_size,num_of_features))
#         #ml_model = tf.keras.layers.GRU(256, return_sequences=True)(inputs)
#         ml_model = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)
#         flat = Flatten()(ml_model)
#         outputs = Dense(4, activation='softmax')(flat)
#         model = tf.keras.Model(inputs, outputs)
        
        ######
        #sample size:38252/524, accuracy: 1.0000 - val_loss: 5.5889 - val_accuracy: 0.4403
        #sample size:38252/73, loss: 1.4733e-04 - accuracy: 1.0000 - val_loss: 4.6835 - val_accuracy: 0.5475
        model = Sequential()
        model.add(LSTM(256, return_sequences=True, input_shape=(sample_size,num_of_features)))

        model.add(Flatten())
        model.add(Dense(4))
        model.add(Activation('softmax'))        
        
        ######
#         model = Sequential()
#         model.add(LSTM(256, return_sequences=True, input_shape=(sample_size,num_of_features), go_backwards=True))
#         model.add(Flatten())
#         model.add(Dense(4))
#         model.add(Activation('softmax'))

        ######
        
#         model = Sequential()
#         model.add(Bidirectional(LSTM(256, return_sequences=True), 
#                                 input_shape=(sample_size,num_of_features))) #, merge_mode='concat'))
#         model.add(Flatten())
#         model.add(Dense(4))
#         model.add(Activation('softmax')) 
        

        model.summary()
        tf.keras.utils.plot_model(model)

        opt_adam = keras.optimizers.Adam(learning_rate=0.001)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(save_to + '_best_model_lstm_time_domain.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
            
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * np.exp(-epoch / 10.))
            
        model.compile(optimizer=opt_adam,
                      loss=['categorical_crossentropy'],
                      metrics=['accuracy'])

          
    history = model.fit(x_train,y_train,
                        batch_size=32,
                        epochs=epoch,
                        validation_data=(x_test,y_test),
                        callbacks=[es,mc,lr_schedule], shuffle=False)
        
    # saved_model = load_model(save_to + '_best_model_lstm_all_cat.h5')
        
    return model,history


# In[14]:



model,history = train_model(X_train, y_train,X_test, y_test, save_to= './', epoch = 40, 
                            sample_size=sample_size, num_of_features=num_of_features)


# In[ ]:




