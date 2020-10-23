import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
import matplotlib.pyplot as plt
import csv

x_train = []
y_train = []
#opening csv
with open('dados.csv',newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    line = 0
    for row in reader:
        if line > 1:
            xrow = float(row['x'])
            y_row = float(row['y'])
            x_train.append(xrow)
            y_train.append(y_row)
        line += 1

data_x = np.asarray(x_train)
data_y = np.asarray(y_train)

x_train, x_val = np.split(data_x, 2)
y_train, y_val = np.split(data_y, 2)

model = keras.models.Sequential()

model.add(keras.layers.Dense(1))
model.add(keras.layers.Dense(32, activation=tf.nn.tanh))
model.add(keras.layers.Dense(32, activation=tf.nn.tanh))
model.add(keras.layers.Dense(32, activation=tf.nn.tanh))
model.add(keras.layers.Dense(32, activation=tf.nn.tanh))
model.add(keras.layers.Dense(32, activation=tf.nn.tanh))
model.add(keras.layers.Dense(32, activation=tf.nn.tanh))
model.add(keras.layers.Dense(1, activation=tf.nn.tanh))

model.compile(optimizer='SGD',loss='mse', metrics=['mse'])

history = model.fit(x_train, y_train,  epochs=100, validation_data = (x_val, y_val))


#  "Accuracy"
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# "Loss"
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# validation_arr = model.evaluate(x_val, y_val)
# print("Validation loss: " + str(validation_arr[0]) + "\nValidation accuracy: " + str(validation_arr[1]))


