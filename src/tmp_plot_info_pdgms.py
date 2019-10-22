#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fr July 12 18:03:28 2018

@author: TDA-Team:  Sebastian Dobrzynski, Daniel Gonzalez, Andre Schulze

This script computes was created in order to plot the results of the persistent diagrams.

"""

# Add pershombox module to path.
from skimage import filters
import os
import sys
sys.path.insert(1, '/home/daniel/Studies/Uni-Heidelberg/TDA/c-hofer/tda-toolkit')

# Load dependencies
import numpy as np
import matplotlib.pyplot as plt
#from data_prep import * 
from pershombox import calculate_discrete_NPHT_2d, distance_npht2D


X_300 = np.load("../data/data_set/X_300.npy")
y_300 = np.load("../data/data_set/y_300.npy") # Labels of the training data.
pdgms_300_4angl = np.load("../data/pdgms/pdgms_300_4angl.npy")
pdgms_300_4angl_list = np.ndarray.tolist(pdgms_300_4angl)

for i in range(len(pdgms_300_4angl_list)):
    print("label = {}".format(y_300[i]))

    #for j in range(4):
    print(pdgms_300_4angl_list[i][1][1])
    img = X_300[i].reshape(8,8)
    val = filters.threshold_otsu(img)
    img2 = img <= val 

    print(img)
    fig = plt.figure(figsize = (7,7))
    plt.gray()
    plt.subplot('121'); plt.axis('off')
    plt.imshow(img, interpolation = "nearest")
    plt.subplot('122'); plt.axis('off')
    plt.imshow(img2, interpolation = "nearest")
    fig.tight_layout(); plt.show()
