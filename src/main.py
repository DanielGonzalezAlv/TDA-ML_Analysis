#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fr July 12 18:03:28 2018

@author: TDA-Team:  Sebastian Dobrzynski, Daniel Gonzalez, Andre Schulze

main script for TDA-ML analysis
"""

#%% Load external dependencies
import numpy as np
from numpy import linalg as LA
from sklearn import svm

def k_sigma(F, G, sigma):
    """Kernel on the space of persistent diagrams.

    Parameters
    ----------
    F : list
        Persistent diagram for one filtration and homology group: A list of 2-tuples.
    G : list
        Same as F.
    sigma: real
        scale, > 0

    Returns
    -------
    real
        Inner product of F, G in the feature space.
    """
    const = -8 * sigma
    sum = 0
    for y in F:
        for z in G:
            sum += np.exp(((y[0]-z[0])**2 + (y[1]-z[1])**2) / const)
            sum -= np.exp(((y[0]-z[1])**2 + (y[1]-z[0])**2) / const)

    return sum / (8 * np.pi * sigma)

def k_sigma_filt(F, G, sigma):
    """Kernel adapted to sets of filtrations and homology groups."""
    sum = 0
    for filt in range(len(F)): # Iterate over filtrations.
        for dim in range(len(F[filt])): # Iterate over homology dimensions.
            sum += k_sigma(F[filt][dim], G[filt][dim], sigma)

    return sum

def d_sigma(F, G, sigma):
    """Pseudo-metric"""
    return np.sqrt(k_sigma(F, F, sigma) + k_sigma(G, G, sigma) - 2*k_sigma(F, G, sigma))

def gram_matrix(Fs, Gs, k):
    """Gram matrix"""
    G = np.empty((len(Fs), len(Gs)))
    for idF in range(len(Fs)):
        for idG in range(len(Gs)):
            G[idF][idG] = k(Fs[idF], Gs[idG])

    return G

#%%
if __name__ == "__main__":
    # X_test = np.load("../data/pdgms/X_test.npy")
    # X_train = np.load("../data/pdgms/X_train.npy")

    y_train = np.load("../data/pdgms/y_train.npy") # Labels of the training data.
    y_test = np.load("../data/pdgms/y_test.npy") # Labels of the test data.
    pdgms_train = np.ndarray.tolist(np.load("../data/pdgms/pdgms_train.npy", allow_pickle=True)) # Persistent diagrams of the training data.
    pdgms_test = np.ndarray.tolist(np.load("../data/pdgms/pdgms_test.npy", allow_pickle=True)) # Persistent diagrams of the test data.
    
    k_1_filt = lambda F, G: k_sigma_filt(F, G, 1)
    gram = lambda F, G: gram_matrix(F, G, k_1_filt)

    # train
    G_train = gram(pdgms_train, pdgms_train)
    clf = svm.SVC(kernel="precomputed")
    clf.fit(G_train, y_train)
    # print(clf.score(G_train, y_train))

    # predict
    G_predict = gram(pdgms_test, pdgms_train)
    y_pred = clf.predict(G_predict)
    result = np.array(y_pred) == np.array(y_test)
    acc = np.ndarray.tolist(result).count(True) / len(result)

    # TODO: SVM does not like the data layout of pdgms.
    # clf = svm.SVC(kernel=gram)
    # clf.fit(pdgms, y_test)

    # test predictions

#%%
