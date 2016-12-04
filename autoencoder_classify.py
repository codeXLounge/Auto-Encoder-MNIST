import cPickle
import gzip
import os
import numpy as np
from numpy import random as rng
import scipy
import load_data
from sklearn import svm 

hidden_size = 120
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

def classify(train_data,train_lables,test_data, weights):
	W, b1, b2 = weights
	hidden = np.dot(train_data, np.transpose(W))+b1
	hidden =  sigmoid(hidden)
	print hidden.shape

	SVM = svm.LinearSVC()
	SVM.fit(train_data, train_labels)
	predicted_labels = SVM.predict(test_data)
	accuracy = 100.0 * np.sum(predicted_labels==test_labels)/test_labels.shape[0]
	print accuracy

# transformInput(test_data, [W, b1, b2])
classify(train_data,train_labels, test_data,[W,b1,b2])

