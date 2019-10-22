#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fr July 12 18:03:28 2018

@author: TDA-Team:  Sebastian Dobrzynski, Daniel Gonzalez, Andre Schulze

This script loads the data, shuffle it, and generate traing/test 
data set.
Because we want to have the hole data in the same file, we merge everything together at the end.
"""

# Load dependencies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import model_selection

def plot_image(img, img_nr):
    """
    This function is intended to plot digits for ilustration
    """
    img = img[img_nr,:,:]
    fig = plt.figure(figsize = (7,7))
    plt.gray()
    plt.subplot('131'); plt.axis('off')
    plt.imshow(img, interpolation = "nearest")
    plt.subplot('132'); plt.axis('off')
    plt.imshow(img, interpolation = "gaussian")
    plt.subplot('133'); plt.axis('off')
    plt.imshow(img, interpolation = "spline36")
    fig.tight_layout(); plt.show()

def prep_MNIST():
    """
    :return: 90% training-dataset, 10% test-dataset
    """
    ## load digits data set
    digits = load_digits()
    data = digits["data"]
    images = digits["images"]
    target = digits["target"]
    target_names = digits["target_names"]
    # print("Data shape - Data type: ", data.shape, data.dtype)

    # split into train and test data
    X_all = data
    y_all = target
    X_train, X_test, y_train , y_test = \
        model_selection.train_test_split(
                digits.data, digits.target,
                test_size = 0.4, 
                random_state = 0)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train , y_test= prep_MNIST()
    X_all = np.append(X_train, X_test, axis= 0)
    y_all = np.append(y_train, y_test, axis=0)
    X_300 = np.append(X_train[0:240], X_test[0:60], axis=0)
    y_300 = np.append(y_train[0:240], y_test[0:60], axis=0)
    
    # Save npy files
    np.save("../data/data_set/X_train", X_train)
    np.save("../data/data_set/y_train", y_train)
    np.save("../data/data_set/X_test", X_test)
    np.save("../data/data_set/y_test", y_test)
    np.save("../data/data_set/X_300", X_300)
    np.save("../data/data_set/y_300", y_300)
    np.save("../data/data_set/X_all", X_all)
    np.save("../data/data_set/y_all", y_all)
