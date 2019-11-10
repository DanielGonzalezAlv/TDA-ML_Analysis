#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fr July 12 18:03:28 2018

@author: TDA-Team:  Sebastian Dobrzynski, Daniel Gonzalez, Andre Schulze

"""

#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'src'))
	print(os.getcwd())
except:
	pass

#%%
import math as mt
import numpy as np

#input: a normal list
def alg_appr(dg_1, dg_2, m, sigma):
    
    theta = -mt.pi/2
    s = mt.pi/m

    v_1 = np.array([])
    v_2 = np.array([])
    dp_1 = [None]*len(dg_1)        #projections
    dp_2 = [None]*len(dg_2)
    sli_was = 0
    
    for i, p_k in enumerate(dg_1):
        dp_1[i] = ((p_k[0]+p_k[1])/2, (p_k[0]+p_k[1])/2)
    for i, p_k in enumerate(dg_2):
        dp_2[i] = ((p_k[0]+p_k[1])/2, (p_k[0]+p_k[1])/2)
    
    df_1 = np.asarray(dg_1 + dp_2)
    df_2 = np.asarray(dg_2 + dp_1)
    
    for i in range(m):
        theta_vec = np.array([mt.cos(theta), mt.sin(theta)])
        for p_k in df_1:
            v_1 = np.append(v_1, np.dot(theta_vec, p_k))
        for p_k in df_2:
            v_2 = np.append(v_2, np.dot(theta_vec, p_k))
        w_1 = np.sort(v_1)
        w_2 = np.sort(v_2)
        sli_was = sli_was + s*np.linalg.norm(w_1 - w_2, ord=1)
        theta += s
        v_1 = np.array([])
        v_2 = np.array([])

    return np.exp(sli_was / (-2 * mt.pi * sigma*sigma))

def alg_appr_filt(dg_1, dg_2, m, sigma, maxDim):
    """Kernel adapted to sets of filtrations and homology groups."""
    sum = 0
    for filt in range(len(dg_1)): # Iterate over filtrations.
        for dim in range(min(len(dg_1[filt]), maxDim)): # Iterate over homology dimensions.
            sum += alg_appr(dg_1[filt][dim], dg_2[filt][dim], m, sigma)

    return sum

#%%
