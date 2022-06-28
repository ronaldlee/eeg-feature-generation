
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import mutual_info_classif
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks,layers

from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional,BatchNormalization,PReLU,ReLU,Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model


init_df = pd.read_csv('csv/out_gameemo.csv',  sep=',')

label_map = {1:"HA_PV", 2:"HA_NV", 3:"LA_NV", 4:"LA_PV"}

init_df["Label"] = init_df["Label"].map(label_map)

features = init_df.iloc[:, :-1]
label = init_df.iloc[:, -1:]

y = label
X = features

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=48)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


model=keras.Sequential([
    layers.Dense(units=3445,input_shape=(3445,),activation='relu'), #using the relu activation because it is great for hidden layers
    layers.BatchNormalization(), #BatchNormalization layer scales the dataset even further
    layers.Dropout(0.27), #Dropping-out the nodes to make our model more general
    layers.Dense(units=3181,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),  
    layers.Dense(units=4181,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.32),  
    layers.Dense(units=2581,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.27),  
    layers.Dense(units=2381,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.32),  
    layers.Dense(units=2181,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.27),  
    layers.Dense(units=4,activation='softmax') #Softmax activation helps in multiclass-identification
])


# Compiling the model
adam=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
#adam=keras.optimizers.Adamax(learning_rate=0.00085, beta_1=0.9, beta_2=0.999, epsilon=1e-07) #These are just general code. you can find them easily in tensorflow API guide
#Categorical_crossentropy will make sure if all the categories are getting identified
#Accuracy will help in identifying if correct labels are getting picked-up
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

# Fitting the model
es=callbacks.EarlyStopping(patience=10,min_delta=0.0001,restore_best_weights=True)

mc=ModelCheckpoint('./_best_model_gameemo_mlpnn.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


# Defining earlystopping callback to save time.
history=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=32,epochs=40,callbacks=[es,mc])


        
