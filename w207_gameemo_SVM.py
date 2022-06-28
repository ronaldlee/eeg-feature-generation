#https://keras.io/examples/keras_recipes/quasi_svm/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional,BatchNormalization,PReLU,ReLU,Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
 

init_df = pd.read_csv('csv/out_gameemo.csv',  sep=',')
label_map = {1:"HA_PV", 2:"HA_NV", 3:"LA_NV", 4:"LA_PV"}

init_df["Label"] = init_df["Label"].map(label_map)

features = init_df.iloc[:, :-1]
label = init_df.iloc[:, -1:]


y = label
X = features

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=48)

#y_train = keras.utils.to_categorical(y_train)
#y_test = keras.utils.to_categorical(y_test)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


model = keras.Sequential(
    [
        keras.Input(shape=(3445)),
        RandomFourierFeatures(
            output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        ),
        layers.Dense(units=4),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.hinge,
    metrics=['accuracy']
)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('./_best_model_svm_gameemo.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
 
model.fit(X_train, y_train, epochs=40, batch_size=32, 
        validation_data=(X_test,y_test),callbacks=[es,mc])


