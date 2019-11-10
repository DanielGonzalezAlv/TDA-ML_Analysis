#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fr July 19 12:43:08 2018
@author: TDA-Team:  Sebastian Dobrzynski, Daniel Gonzalez, Andre Schulze
"""

#%% Load external dependencies
import numpy as np
import matplotlib.pyplot as plt
import parallel_joblib_counter as ctr
import os
from numpy import linalg as LA
from sklearn import svm
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

import reininghaus_2015_kernel as _2015
import _1706_approx_kernel as _1706

#%%
def gram_matrix(Fs, Gs, k, noisy=False):
    """Gram matrix"""

    if noisy: # print progress
        print("progress gram matrix")
        step = len(Fs) * len(Gs) // 100
        manager = ctr.Manager()
        counter = ctr.Counter(manager, 0)
        def k_ctr(x, y, ctr_):
            ctr_.increment()
            if ctr_.value() % step == 0:
                print(".".format(os.getpid()))
            return k(x, y)
        G = Parallel(n_jobs=12)(delayed(k_ctr)(Fs[idF], Gs[idG], counter)
            for idF in range(len(Fs))
            for idG in range(len(Gs))
        )
    else:
        G = Parallel(n_jobs=12)(delayed(k)(Fs[idF], Gs[idG])
            for idF in range(len(Fs))
            for idG in range(len(Gs))
        )

    return np.asarray(G).reshape((len(Fs), len(Gs)))

def cross_validation(kernel, data, labels, k):
    """K-fold cross-validation
    
    Parameters
    ----------
    kernel : function pointer
        Kernel that takes two persistence diagrams as an input.
    data : numpy array
        Data set of persistence diagrams.
    labels : numpy array
        Labels for data.
    k: integer
        Number of partitions for cross-validation.

    Returns
    -------
    list
        List of test scores for each cross validation step.
    """

    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    scores = np.array(())

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]

        s = score(kernel, X_train, X_test, Y_train, Y_test)
        scores = np.append(scores, s)

    return scores

def score(kernel, X_train, X_test, Y_train, Y_test):
    """Scoring (accuracy on test set) of a kernel.
    
    Parameters
    ----------
    kernel : function pointer
        Kernel that takes two persistence diagrams as an input.
    X_train : numpy array
        Training set.
    X_test : numpy array
        Test set.
    Y_train : numpy array
        Training set labels.
    Y_test : numpy array
        Test set labels.

    Returns
    -------
    list
        List of test scores for each cross validation step.
    """
    gram = lambda F, G: gram_matrix(F, G, kernel)
    clf = svm.SVC(kernel="precomputed")
    clf.fit(gram(X_train, X_train), Y_train)

    Y_pred = clf.predict(gram(X_test, X_train))
    result = np.array(Y_pred) == np.array(Y_test)

    return np.ndarray.tolist(result).count(True) / len(result)

def plot_convergence(kernel, pdgms, labels, samples):
    # convergence plot
    x = [None] * samples # x-axis
    scores = [None] * samples

    # Split the set into samples + 1 partitions and use the last one as the test set.
    split = samples * len(pdgms) // (samples + 1)
    train_set = np.array(pdgms[0 : split])
    test_set = np.array(pdgms[split : len(pdgms)])
    gram = gram_matrix(train_set, train_set, kernel)
    clf = svm.SVC(kernel="precomputed")

    for i in range(samples):
        size = (i + 1) * len(pdgms) // (samples + 1)

        clf.fit(gram[np.ix_(range(size), range(size))], labels[: size])

        pred = clf.predict(gram_matrix(test_set, train_set[: size], kernel))
        result = np.array(pred) == np.array(labels[split : len(pdgms)])

        x[i] = size
        scores[i] = np.ndarray.tolist(result).count(True) / len(result)
    
    plt.plot(x, scores)
    plt.xlabel('training size')
    plt.ylabel('accuracy')
    axes = plt.gca()
    axes.set_ylim([0, 1])


    return (x, scores)

#%%
if __name__ == "__main__":
    y = np.load("../data/data_set/y_all.npy") # Labels

    # Persistent diagrams (different filtrations and different input images (gray and binary))
    pdgms_8angl = np.ndarray.tolist(np.load("../data/pdgms/pdgms_all_8angl.npy", allow_pickle=True))
    pdgms_4angl = np.ndarray.tolist(np.load("../data/pdgms/pdgms_all_4angl.npy", allow_pickle=True))
    #pdgms_8angl = pdgms_8angl[0:200]

    maxDim = 1 # Maximum homology dimension used for computation.
    #k_2015 = lambda F, G: _2015.k_sigma_filt(F, G, 0.001, maxDim)
    k_1706 = lambda F, G: _1706.alg_appr_filt(F, G, 5, 0.5, maxDim)

    #k = 2 # K-fold cross-validation
    #scores_2015 = cross_validation(k_2015, np.array(pdgms_8angl), y, k)
    #scores_1706 = cross_validation(k_1706, np.array(pdgms_8angl), y, k)

    samples = 10
    scores = plot_convergence(k_1706, pdgms_4angl, y, samples)
    
    np.save("../data/plots/k_1706_4angl", scores)
#%%
