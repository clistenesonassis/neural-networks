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

def time_series(n):
    x = math.sin(n + math.sin(n)**2)
    return x

for i in range(0,2000):
    x = [time_series(i - 1), time_series(i - 2), time_series(i - 3), time_series(i - 4), time_series(i - 5)]
    x_train.append(x)
    
    y1 = time_series(i + 1)
    y_train_1step.append(y1)
    y2 = time_series(i + 2)
    y_train_2step.append(y2)
    y3 = time_series(i + 3)
    y_train_3step.append(y3)




# #opening csv
# with open('dados.csv',newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     line = 0
#     for row in reader:
#         if line > 1:
#             xrow = float(row['x'])
#             y_row = float(row['y'])
#             x_train.append(xrow)
#             y_train.append(y_row)
#         line += 1

data_x = np.asarray(x_train)
# data_y = np.asarray(y_train_1step)
# data_y = np.asarray(y_train_2step)
data_y = np.asarray(y_train_3step)

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

