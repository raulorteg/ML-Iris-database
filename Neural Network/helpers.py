"""
Created on Wed Aug 19 19:32:44 2020

@author: Raul Ortega Ochoa
"""

import numpy as np
#============================================================================

# given z, either scalar or array, compute sigmoid for each element
# then return array or scalar
def sigmoid(z):
    return 1/(np.ones((len(z),1)) + np.exp(-z))

#============================================================================

# given z compute sigmoid'(z) (derivative of sigmoid)
def derSigmoid(z):
    return np.multiply(sigmoid(z),(np.ones((len(z), 1)) - sigmoid(z)))

#============================================================================

# I need to reshape y_train so that can be compare with the outputs of the NN
def reshape_class(s):
    types = np.array(['Iris-virginica', 'Iris-setosa', 'Iris-versicolor'])
    foo = types == s
    return np.reshape(foo.astype(np.int), (len(types), 1))

#============================================================================

# given an arrya where every element is the probability return the class with 
# highest probability
def predict_Iris(p_array):
    types = np.array(['Iris-virginica', 'Iris-setosa', 'Iris-versicolor'])
    return types[np.argmax(p_array)]