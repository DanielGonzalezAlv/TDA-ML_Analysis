#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fr July 12 18:03:28 2018

@author: TDA-Team:  Sebastian Dobrzynski, Daniel Gonzalez, Andre Schulze

main script for TDA-ML analysis
"""

# Load external dependencies
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_test = np.load("../data/pdgms/X_test.npy")
    y_test = np.load("../data/pdgms/y_test.npy")
    X_train = np.load("../data/pdgms/X_train.npy")
    y_train = np.load("../data/pdgms/y_train.npy")
    pdgms = np.load("../data/pdgms/pdgms.npy")
    pdgms = np.ndarray.tolist(pdgms)

