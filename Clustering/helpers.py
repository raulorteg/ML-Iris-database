"""
Created on Wed Aug 19 09:57:06 2020

@author: Raul ortega Ochoa
"""

import numpy as np

# given a sample x^(i) compute the distance to every centroid mu_k and return 
# the label of the closest centroid c^(i) = k for that sample
def assignCentroid(x, mu):
    min_sqdistance = np.inf
    for i in range(len(mu)):
        sqdistance = np.matmul(x-mu[i],np.transpose(x-mu[i]))
        if sqdistance < min_sqdistance:
            min_sqdistance = sqdistance
            c = i
    return c, min_sqdistance

#============================================================================