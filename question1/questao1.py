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
            xrow = [float(row['c1']),float(row['c2']),float(row['c3'])]
            y_row = int(row['y'])
            x_train.append(xrow)
            y_train.append(y_row)
        line += 1

data_x = np.asarray(x_train)
data_y = np.asarray(y_train)

x_train, x_val = np.split(data_x, 2)
y_train, y_val = np.split(data_y, 2)

model = keras.models.Sequential()

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(8, activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=25)

validation_arr = model.evaluate(x_val, y_val)
print("Validation loss: " + str(validation_arr[0]) + "\nValidation accuracy: " + str(validation_arr[1]))


