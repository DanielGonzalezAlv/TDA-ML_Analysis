#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fr July 12 18:03:28 2018

@author: TDA-Team:  Sebastian Dobrzynski, Daniel Gonzalez, Andre Schulze

This script computes the presistent diagrams using the approach
of @c-hofer (angles filtrations)
"""

# Load dependencies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import PIL
from persim import plot_diagrams
from ripser import ripser, lower_star_img
from data_prep import * 

## c-hofer-libraries
import sys
sys.path.insert(1, '/home/daniel/Studies/Uni-Heidelberg/TDA/c-hofer/tda-toolkit')# /pershombox/')
#import pershombox
from pershombox import calculate_discrete_NPHT_2d, distance_npht2D


if __name__ == "__main__":
    X_train, X_test, y_train , y_test = prep_MNIST_50()
    i = 0  
    #pdgms = calculate_discrete_NPHT_2d(X_train[i],4)
    
    pdgms = []
    for i in range(X_train.shape[0]):
        img = X_train[i].reshape(8,8)
        pdgms.append(calculate_discrete_NPHT_2d(img,4))

    pdgms = np.asarray(pdgms)

    np.save("../data/pdgms/X_test", X_test)
    np.save("../data/pdgms/X_train", X_train)
    np.save("../data/pdgms/y_train", y_train)
    np.save("../data/pdgms/y_test", y_test)
    np.save("../data/pdgms/pdgms", pdgms)
