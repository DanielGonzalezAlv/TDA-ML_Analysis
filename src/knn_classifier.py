#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fr July 19 12:43:08 2018
@author: TDA-Team:  Sebastian Dobrzynski, Daniel Gonzalez, Andre Schulze

This script ist intended for the TDA_ML_analysis project.  
It is based on a sample solution of an excerse sheet of the course
Fundamentals of Machine Learning hold in WS 2018 in Heidelberg
"""

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits

# load digits data set
digits = load_digits()

data = digits["data"]
images = digits["images"]
target = digits["target"]
target_names = digits["target_names"]

img = images[3,:,:]

# split into train and test data
#X_all = data
#y_all = target
#X_train, X_test, y_train , y_test = model_selection.train_test_split(
#                                    digits.data,
#                                    digits.target,
#                                    test_size = 0.4,
#                                    random_state = 0)
#print("X_train.shape = {}, X_test.shape = {}".format(X_train.shape, X_test.shape))

X_all = np.load("../data/data_set/X_all.npy")
y_all = np.load("../data/data_set/y_all.npy") # Labels of the training data.
X_train = X_all[163:]
X_test = X_all[0:163]
y_train = y_all[163:]
y_test = y_all[0:163]

# euclidean distance computation using broadcasting
def dist_vec(training, test):
    return np.sqrt(np.sum(np.square(np.expand_dims(training, axis = 1) - np.expand_dims(test, axis = 0)),axis = 2))


# Implement the k-nearest neighbour classifer
# k-nearest-neighbor classifier based on vectorized distance computation
def k_nearest_neighbors(X_train, y_train, X_test, k):
    # find k nearest neighbors for each test image
    dist = dist_vec(X_train, X_test)
    indices = np.argsort(dist, axis = 0)[:k,:] # 2D array containing indices of k neighbors with lowest distance
    neighbors = y_train[indices] # indexing y_train (1D) with indices (2D) gives 2D array of neighbors' classes!
    # find most common class among neighbors of each test image
    # (you could also use np.bincount() or scipy.stats.mode() for the voting)
    counts = np.zeros((10, X_test.shape[0]))
    for i in range(k):
        for j in range(X_test.shape[0]):
            counts[neighbors[i,j], j] += 1

    return np.argmax(counts, axis = 0)

# Compute predictions
steps = [1.63000000e+02, 3.26000000e+02, 4.90000000e+02, 6.53000000e+02,
        8.16000000e+02, 9.80000000e+02, 1.14300000e+03, 1.30600000e+03,
        1.47000000e+03, 1.63300000e+03]
scores = np.empty([2,len(steps)])
scores[0] = np.array(steps)
k = 1
i=0
for s in steps:
    knn_prediction = k_nearest_neighbors(X_train[0:int(s)], y_train[0:int(s)], X_test, k)
    accuracy_knn = sum(knn_prediction == y_test)/y_test.shape[0]*100
    scores[1,i] = accuracy_knn
    print("accuracy knn = {}".format(accuracy_knn))
    i = i+1

np.save("../data/plots/knn_1", scores)

   # # Print classification errors
   # pred_errors = np.where( (knn_prediction == y_test) == False)[0]
   # print("Errors in knn = {}".format(y_test[pred_errors]))
    
