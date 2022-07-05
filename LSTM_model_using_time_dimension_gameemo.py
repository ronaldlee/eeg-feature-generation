#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional,BatchNormalization,PReLU,ReLU,Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical




# In[96]:



init_df = pd.read_csv('./csv/out_gameemo_time_domain_simple.csv',  sep=',')

print('Shape of data: ', init_df.shape)


# In[97]:



df = init_df.copy()
print(df.head())

#HA_PV = high arousal, positive valence
#HA_NV = high arousal, negative valence
#LA_NV = low arousal, negative valence
#LA_PV = low arousal, positive valance
label_map = {1:"HA_PV", 2:"HA_NV", 3:"LA_NV", 4:"LA_PV"}

df["Label"] = df["Label"].map(label_map)

# df = df.to_numpy()


# In[ ]:


# Restructure the X features data set to group them by samples.
# We know the sample size is 38252 each, so we just need to iterate and group them




# In[135]:





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

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=48)

total_samples_count = int(X.shape[0]/38252)

print("total_samples_count:", total_samples_count)


train_sample_count = int(total_samples_count * 0.7)
test_sample_count = total_samples_count - train_sample_count

train_size = train_sample_count * 38252
test_size = test_sample_count * 38252

print("train size:", train_size)
print("test size:", test_size)

X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]


X_train = np.array(X_train).reshape((train_sample_count,38252,14))
X_test = np.array(X_test).reshape((test_sample_count,38252,14))

print("X_train.shape after reshape:",X_train.shape)
print("X_test.shape after reshape:",X_test.shape)

#collapse y_train and y_test to the same X sample counts instead

y_train_collapsed = np.array([])
for i in range(len(y_train)):
    if (i % 38252 == 0):
        y_train_collapsed = np.append(y_train_collapsed, (y_train.iloc[i]))
        
print("y_train_collapsed shape:",y_train_collapsed.shape)        

y_test_collapsed = np.array([])
for i in range(len(y_test)):
    if (i % 38252 == 0):
        y_test_collapsed = np.append(y_test_collapsed, (y_test.iloc[i]))
        
print("y_test_collapsed shape:",y_test_collapsed.shape)    


y_train = pd.get_dummies(y_train_collapsed)
y_test = pd.get_dummies(y_test_collapsed)

print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

 


# In[141]:


def train_model(x_train, y_train,x_test,y_test, save_to, epoch = 2):
    strategy = tf.distribute.MirroredStrategy(devices=None)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        #inputs = tf.keras.Input(shape=(X_train.shape[0],14)) #input_dim = 14 channels(features)
#         inputs = tf.keras.Input(shape=(38252,14)) #input_dim = 14 channels(features)
        inputs = tf.keras.Input(shape=(38252,14))
        

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


# In[142]:



model,history = train_model(X_train, y_train,X_test, y_test, save_to= './', epoch = 40)


# In[ ]:




