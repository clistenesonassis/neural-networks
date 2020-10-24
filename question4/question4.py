import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
import matplotlib.pyplot as plt
import csv
import math

x_train = []
y_train_1step = []
y_train_2step = []
y_train_3step = []

step = 0.1
steps = 10
samples = 2000

def time_series(n):
    x = math.sin(n + math.sin(n)**2)
    return x

def generate_data(step, k, samples):
    x = []
    for i in range(0, samples):
        for j in range(1, k):
                x.append(time_series(i*step - j*step))
        x_train.append(x)

        y_train_1step.append(time_series(i*step + step))
        y_train_2step.append(time_series(i*step + 2*step))
        y_train_3step.append(time_series(i*step + 3*step))

        x = []

generate_data(step, steps, samples)


data_x = np.asarray(x_train)
data_y = np.asarray(y_train_1step)
# data_y = np.asarray(y_train_2step)
# data_y = np.asarray(y_train_3step)

x_train, x_val = np.split(data_x, 2)
y_train, y_val = np.split(data_y, 2)

model = keras.models.Sequential()

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(8, activation=tf.nn.tanh))
model.add(keras.layers.Dense(8, activation=tf.nn.tanh))
model.add(keras.layers.Dense(8, activation=tf.nn.tanh))
model.add(keras.layers.Dense(8, activation=tf.nn.tanh))
model.add(keras.layers.Dense(1, activation=tf.nn.tanh))

model.compile(optimizer='adam',loss='mse', metrics=['mse'])

history = model.fit(x_train, y_train,  epochs=30, validation_data = (x_val, y_val))# "Loss"

#loss
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

real = []
predito = []

predicted = model.predict(x_val)
rangex = int(12/step)
for j in range(rangex):
    real.append(y_val[j])
    predito.append(predicted[j])

plt.plot(real) 
plt.plot(predito) 
plt.title('Predição vs real') 
plt.legend(['real', 'predição'], loc='upper left') 
plt.show()