import cPickle
import gzip
import os
    
import numpy as np
from numpy import random as rng
import scipy
import load_data

hidden_size = 30
# testing part
weights = np.load(str(hidden_size)+'_weights.npz')
W = weights['W']
b1 = weights['b1']
b2 = weights['b2']

train, valid, test  = load_data.load_data('/home/sensors/Desktop/neural/mnist.pkl.gz')
train_data = train[0]
train_labels = train[1]
test_data = test[0]
test_labels = test[1]

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def toimg(X, name):
	Y = np.reshape(X, (-1,28))
	scipy.misc.imsave(name,Y)


def transformInput(data, weights):
	W, b1, b2 = weights
	hidden = np.dot(data, np.transpose(W))+b1
	hidden =  sigmoid(hidden)
	output = np.dot(hidden, (W)) + b2
	output = sigmoid(output)
	# print W.shape
	# print data.shape
	# print b1.shape
	# print b2.shape
	# print hidden.shape
	# print output.shape

	toimg(output[0], 'img0_'+str(hidden_size)+'.png')
	toimg(output[1], 'img1_'+str(hidden_size)+'.png')
	toimg(output[2], 'img2_'+str(hidden_size)+'.png')
	toimg(output[3], 'img3_'+str(hidden_size)+'.png')
	toimg(output[4], 'img4_'+str(hidden_size)+'.png')

def visualizeFeatures(data, weights):
	W = weights[0]
	W = np.vsplit(W,1)
	for i in range(0, hidden_size):
		feature = np.multiply(W[0][i], data[0])
		toimg(feature, 'feature0_'+str(hidden_size)+'_'+str(i)+'.png')

	# print W[0][1]
	# print data[0]



# transformInput(test_data, [W, b1, b2])
visualizeFeatures(test_data,[W,b1,b2])

# print type(train_data), train_data.shape
# print type(train_labels), train_labels.shape
# print type(test_data), test_data.shape
# print type(test_labels), test_labels.shape