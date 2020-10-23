import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
import matplotlib.pyplot as plt
import csv

# def switch_demo(argument):
#     switcher = {
#         0: [1, -1, -1, -1, -1, -1, -1, -1],
#         1: [1, 1, -1, -1, -1, -1, -1, -1],
#         2: [1, -1, 1, -1, -1, -1, -1, -1],
#         3: [1, -1, -1, 1, -1, -1, -1, -1],
#         4: [1, -1, -1, -1, 1, -1, -1, -1],
#         5: [1, -1, -1, -1, -1, 1, -1, -1],
#         6: [1, -1, -1, -1, -1, -1, 1, -1],
#         7: [1, -1, -1, -1, -1, -1, -1, 1]
#     }
#     return switcher.get(argument)


x_train = []
y_train = []
#opening csv
with open('dados.csv',newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    line = 0
    for row in reader:
        if line != 0:
            xrow = [float(row['c1']),float(row['c2']),float(row['c3'])]
            y_row = int(row['y'])
            x_train.append(xrow)
            y_train.append(y_row)
        line += 1


##mnist = keras.datasets.mnist

## x_train = [[0,0],[0,1],[1,0],[1,1]]
# y_train = [0,1,2,3]
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_train = keras.utils.normalize(x_train, axis=1)
# # # x_test = keras.utils.normalize(x_test, axis=1)

model = keras.models.Sequential()

model.add(keras.layers.Flatten())
# # model.add(keras.layers.Dense(128, activation=tf.nn.relu))
# # model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(8, activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy','mse'])

model.fit(x_train, y_train, epochs=30)

# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)