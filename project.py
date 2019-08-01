#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fr July 12 18:03:28 2018

@author: TDA-Team:  Sebastian Dobrzynski, Daniel Gonzalez, Andre Schulze

This script was created for TDA-ML analysis  
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
## c-hofer-libraries
import sys
sys.path.insert(1, '/home/daniel/Studies/Uni-Heidelberg/TDA/c-hofer/tda-toolkit')# /pershombox/')
#import pershombox
from pershombox import calculate_discrete_NPHT_2d, distance_npht2D

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

if __name__ == "__main__":
    
    ## load digits data set
    digits = load_digits()
    data = digits["data"]
    images = digits["images"]
    target = digits["target"]
    target_names = digits["target_names"]
    # print("Data shape - Data type: ", data.shape, data.dtype)

    ## Plot image as an example
    #plot_image(images, 0)

    # Compute the NPHT for 2d cubical complexes with equidistant directions
    img = images[0,:,:]
    npht_0 = calculate_discrete_NPHT_2d(img,4)






