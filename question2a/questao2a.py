import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
import matplotlib.pyplot as plt

x_train = np.array([ [0,0], [0,1], [1,0], [1,1]])
y_train = np.array([ [0], [1], [1], [0] ])


model = keras.models.Sequential()

model.add(keras.layers.Dense(2))
model.add(keras.layers.Dense(4, activation=tf.nn.sigmoid))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])

history = model.fit(x_train, y_train,  epochs=10000, batch_size=4)


#  "Accuracy"
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
# plt.plot(history.history['val_mse'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# validation_arr = model.evaluate(x_val, y_val)
# print("Validation loss: " + str(validation_arr[0]) + "\nValidation accuracy: " + str(validation_arr[1]))


