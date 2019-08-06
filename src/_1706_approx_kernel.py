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
def alg_appr(dg_1, dg_2, m):
    
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
    
    dg_1 += dp_2
    dg_2 += dp_1
    # np.append(dg_1, dp_2)
    # np.append(dg_2, dp_1)
    
    dg_1 = np.asarray(dg_1)   #want to use dot product in np
    dg_2 = np.asarray(dg_2)
    for i in range(m):
        theta_vec = np.array([mt.cos(theta), mt.sin(theta)])
        for p_k in dg_1:
            v_1 = np.append(v_1, np.dot(theta_vec, p_k))
        for p_k in dg_2:
            v_2 = np.append(v_2, np.dot(theta_vec, p_k))
        w_1 = np.sort(v_1)
        w_2 = np.sort(v_2)
        sli_was = sli_was + s*np.linalg.norm(w_1 - w_2, ord=1)
        theta += s
        v_1 = np.array([])
        v_2 = np.array([])

    return (1/mt.pi)*sli_was

def alg_appr_filt(dg_1, dg_2, m):
    """Kernel adapted to sets of filtrations and homology groups."""
    sum = 0
    for filt in range(len(dg_1)): # Iterate over filtrations.
        for dim in range(len(dg_1[filt])): # Iterate over homology dimensions.
            sum += alg_appr(dg_1[filt][dim], dg_2[filt][dim], m)

    return sum

#%%



