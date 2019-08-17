#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fr July 12 18:03:28 2018

@author: TDA-Team:  Sebastian Dobrzynski, Daniel Gonzalez, Andre Schulze

This script computes the presistent diagrams using the approach
of @c-hofer (angles filtrations)
"""

# Add pershombox module to path.
from skimage import filters
import os
import sys
sys.path.insert(1, '/home/daniel/Studies/Uni-Heidelberg/TDA/c-hofer/tda-toolkit')

# Load dependencies
import numpy as np
from data_prep import * 
from pershombox import calculate_discrete_NPHT_2d, distance_npht2D


X_test = np.load("../data/pdgms/X_test.npy")
X_train = np.load("../data/pdgms/X_train.npy")
y_train = np.load("../data/pdgms/y_train.npy") # Labels of the training data.
y_test = np.load("../data/pdgms/y_test.npy") # Labels of the test data.
pdgms_train = np.ndarray.tolist(np.load("../data/pdgms/pdgms_train.npy", allow_pickle=True)) # Persistent diagrams of the training data.
pdgms_test = np.ndarray.tolist(np.load("../data/pdgms/pdgms_test.npy", allow_pickle=True)) # Persistent diagrams of the test data.

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

for i in range(len(pdgms_train)):
    print("label = {}".format(y_train[i]))
    for j in range(4):
        print(pdgms_train[i][j][1])
        img = X_train[i].reshape(8,8)
        val = filters.threshold_otsu(img)
        img2 = img <= val 

        print(img)
        fig = plt.figure(figsize = (7,7))
        plt.gray()
        plt.subplot('131'); plt.axis('off')
        plt.imshow(img, interpolation = "nearest")
        plt.subplot('132'); plt.axis('off')
        plt.imshow(img, interpolation = "gaussian")
        plt.subplot('133'); plt.axis('off')
        plt.imshow(img2, interpolation = "nearest")
        fig.tight_layout(); plt.show()
