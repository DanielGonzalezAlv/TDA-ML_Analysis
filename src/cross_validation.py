#!/usr/bin/env python3

#%% Load external dependencies
import numpy as np
from numpy import linalg as LA
from sklearn import svm
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

import reininghaus_2015_kernel as _2015
import _1706_approx_kernel as _1706

def gram_matrix(Fs, Gs, k):
    """Gram matrix"""
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

#%%
if __name__ == "__main__":
    y = np.load("../data/data_set/y_300.npy") # Labels

    # Persistent diagrams (different filtrations and different input images (gray and binary))
    pdgms_4angl_thr = np.ndarray.tolist(np.load("../data/pdgms/pdgms_300_4angl_thr.npy", allow_pickle=True))
    pdgms_4angl = np.ndarray.tolist(np.load("../data/pdgms/pdgms_300_4angl.npy", allow_pickle=True))
    pdgms_8angl_thr = np.ndarray.tolist(np.load("../data/pdgms/pdgms_300_8angl_thr.npy", allow_pickle=True))
    pdgms_8angl = np.ndarray.tolist(np.load("../data/pdgms/pdgms_300_8angl.npy", allow_pickle=True))

    maxDim = 1 # Maximum homology dimension used for computation.
    k_2015 = lambda F, G: _2015.k_sigma_filt(F, G, 0.001, maxDim)
    k_1706 = lambda F, G: _1706.alg_appr_filt(F, G, 5, 0.5, maxDim)

    k = 2 # K-fold cross-validation
    scores_2015 = cross_validation(k_2015, np.array(pdgms_4angl), y, k)
    scores_1706 = cross_validation(k_1706, np.array(pdgms_4angl), y, k)

    # Example to score training and test set of different size.
    my_score = score(k_2015, np.array(pdgms_4angl[0:240]), np.array(pdgms_4angl[240:300]),
        y[0:240], y[240:300])

    # TODO: plot data