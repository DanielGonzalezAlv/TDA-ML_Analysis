import numpy as np

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

def k_sigma_filt(F, G, sigma, maxDim):
    """Kernel adapted to sets of filtrations and homology groups."""
    sum = 0
    for filt in range(len(F)): # Iterate over filtrations.
        for dim in range(min(len(F[filt]), maxDim)): # Iterate over homology dimensions.
            sum += k_sigma(F[filt][dim], G[filt][dim], sigma)

    return sum


def d_sigma(F, G, sigma):
    """Pseudo-metric"""
    return np.sqrt(k_sigma(F, F, sigma) + k_sigma(G, G, sigma) - 2*k_sigma(F, G, sigma))