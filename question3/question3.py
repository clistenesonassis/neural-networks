from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from tensorflow import keras as keras
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import random 
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def isInsideCircle(x, y):
  if ((pow(x,2) + pow(y,2)) <= 1):
    return True
  else:
    return False

def area(x1, y1, x2, y2, x3, y3): 
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)  
                + x3 * (y1 - y2)) / 2.0) 
  
def isInsideTriangle(x1, y1, x2, y2, x3, y3, x, y): 
  
    A = area (x1, y1, x2, y2, x3, y3) 
  
    A1 = area (x, y, x2, y2, x3, y3) 
        
    A2 = area (x1, y1, x, y, x3, y3) 
      
    A3 = area (x1, y1, x2, y2, x, y) 

    if(A == A1 + A2 + A3): 
        return True
    else: 
        return False



data_x = []
data_y = []

count = np.zeros(15)


while (len(data_x) < 20000):
  pointX = random.uniform(-1,1)
  pointY = random.uniform(-1,1)

  if (isInsideCircle(pointX, pointY) and pointX != 0 and pointY != 0):
    data_x.append([pointX, pointY])

    if (pointX > 0 and pointY > 0):
      if (isInsideTriangle(0, 0, 0, 1, 1, 0, pointX, pointY)):
        data_y.append(0)
      else:
        data_y.append(4)

    elif (pointX < 0 and pointY > 0):
      if (isInsideTriangle(0, 0, 0, 1, -1, 0, pointX, pointY)):
        data_y.append(1)
      else:
        data_y.append(5)
    elif (pointX < 0 and pointY < 0):
      if (isInsideTriangle(0, 0, 0, -1, -1, 0, pointX, pointY)):
        data_y.append(2)
      else:
        data_y.append(6)
    elif (pointX > 0 and pointY < 0):
      if (isInsideTriangle(0, 0, 0, -1, 1, 0, pointX, pointY)):
        data_y.append(3)
      else:
        data_y.append(7)


colors = ["#FFFF00","#808000","#00FF00","#008000","#00FFFF","#008080","#0000FF","#000080"]

data = np.array(data_x)
x, y = data.T
color_indices = data_y

colormap = matplotlib.colors.ListedColormap(colors)

plt.scatter(x, y, c=color_indices, cmap=colormap)
plt.show()

data_x = np.array(data_x)
data_y = np.array(data_y)
x_train, x_val = np.split(data_x, 2)
y_train, y_val = np.split(data_y, 2)

data_X, x_test, data_Y, y_test = train_test_split(data_x, data_y, test_size=0.20, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size=0.25, random_state=42)

model = keras.models.Sequential()

model.add(keras.layers.Dense(2))
model.add(keras.layers.Dense(8, activation=tf.nn.tanh))
model.add(keras.layers.Dense(8, activation=tf.nn.tanh))
model.add(keras.layers.Dense(8, activation=tf.nn.tanh))
model.add(keras.layers.Dense(8, activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50, validation_data = (x_val, y_val))

# validation_arr = model.evaluate(np.array([[-0.01,-0.2]]), np.array([2]))
# print("Validation loss: " + str(validation_arr[0]) + "\nValidation accuracy: " + str(validation_arr[1]))

plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy']) 
plt.title('model accuracy') 
plt.ylabel('accuracy') 
plt.xlabel('epoch') 
plt.legend(['train', 'validation'], loc='upper left') 
plt.show()

plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('model loss') 
plt.ylabel('loss') 
plt.xlabel('epoch') 
plt.legend(['train', 'validation'], loc='upper left') 
plt.show()


#Pred vs real
prediction=model.predict_classes(x_test)

colors = ["#FFFF00","#808000","#00FF00","#008000","#00FFFF","#008080","#0000FF","#000080"]

data = np.array(x_test)
x, y = data.T
color_indices = prediction

colormap = matplotlib.colors.ListedColormap(colors)

plt.scatter(x, y, c=color_indices, cmap=colormap)
plt.show()

cm = confusion_matrix(y_test, prediction)

print(cm)


df_cm = pd.DataFrame(cm, range(8), range(8))
plt.figure(figsize=(10,7))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g') # font size

plt.show()