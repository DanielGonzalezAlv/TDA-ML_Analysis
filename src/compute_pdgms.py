#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fr July 12 18:03:28 2018

@author: TDA-Team:  Sebastian Dobrzynski, Daniel Gonzalez, Andre Schulze

This script computes the presistent diagrams using the approach
of @c-hofer (angles filtrations) [Reininghaus et al., 2015]
"""

# Add pershombox module to path.
from skimage import filters
import os
import sys
sys.path.insert(1, '/home/daniel/Studies/Uni-Heidelberg/TDA/c-hofer/tda-toolkit')

# Load dependencies
import numpy as np
from arrange_data import * 
from pershombox import calculate_discrete_NPHT_2d, distance_npht2D

X_300 = np.load("../data/data_set/X_300.npy")
y_300 = np.load("../data/data_set/y_300.npy") # Labels of the training data.
X_all = np.load("../data/data_set/X_all.npy")
y_all = np.load("../data/data_set/y_all.npy") # Labels of the training data.

pdgms_300_4angl = [] #for i in range(X_300.shape[0]):
    img = X_300[i].reshape(8,8).copy()
    pdgms_300_4angl.append(calculate_discrete_NPHT_2d(img,4))

# With threshold
pdgms_300_4angl_thr = []
for i in range(X_300.shape[0]):
    img = X_300[i].reshape(8,8).copy()
    val = filters.threshold_otsu(img)
    img[img<val] = 0
    img[img>=val] = 1
    pdgms_300_4angl_thr.append(calculate_discrete_NPHT_2d(img,4))

pdgms_300_8angl = []
for i in range(X_300.shape[0]):
    img = X_300[i].reshape(8,8).copy()
    pdgms_300_8angl.append(calculate_discrete_NPHT_2d(img,8))

# With threshold
pdgms_300_8angl_thr = []
for i in range(X_300.shape[0]):
    img = X_300[i].reshape(8,8).copy()
    val = filters.threshold_otsu(img)
    img[img<val] = 0
    img[img>=val] = 1
    pdgms_300_8angl_thr.append(calculate_discrete_NPHT_2d(img,8))

pdgms_all_4angl = []
for i in range(X_all.shape[0]):
    img = X_all[i].reshape(8,8).copy()
    pdgms_all_4angl.append(calculate_discrete_NPHT_2d(img,4))

pdgms_all_8angl = []
for i in range(X_all.shape[0]):
    img = X_all[i].reshape(8,8).copy()
    pdgms_all_8angl.append(calculate_discrete_NPHT_2d(img,8))

pdgms_300_4angl = np.asarray(pdgms_300_4angl)
pdgms_300_4angl_thr = np.asarray(pdgms_300_4angl_thr)
pdgms_300_8angl = np.asarray(pdgms_300_8angl)
pdgms_300_8angl_thr = np.asarray(pdgms_300_8angl_thr)
pdgms_all_4angl = np.asarray(pdgms_all_4angl)
pdgms_all_8angl = np.asarray(pdgms_all_4angl)

np.save("../data/pdgms/pdgms_300_4angl", pdgms_300_4angl)
np.save("../data/pdgms/pdgms_300_4angl_thr", pdgms_300_4angl_thr)
np.save("../data/pdgms/pdgms_300_8angl", pdgms_300_8angl)
np.save("../data/pdgms/pdgms_300_8angl_thr", pdgms_300_8angl_thr)
np.save("../data/pdgms/pdgms_all_4angl", pdgms_all_4angl)
np.save("../data/pdgms/pdgms_all_8angl", pdgms_all_8angl)
