import numpy as np
from tensorflow import keras
import random
import sys

seed = random.randrange(500000)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigDX(x):
    return x*(1-x)

trai = np.array([[0,0,0],
                [0,0,1],
                [0,1,0]])

trao = np.array([[-1,-0.75,-0.50]]).T

np.random.seed(seed)

wei = 0.2*np.random.random((3,1)) - 1

print('Weights: ')
print(wei)

for i in range(100000):
    ilayer = trai
    outs = sigmoid(np.dot(ilayer, wei))

    error = trao - outs

    adjust = error * sigDX(outs)

    wei += np.dot(ilayer.T, adjust)

print("New weights: ")
print(wei)

print("Outs: ")
print(outs)