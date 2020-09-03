"""
Created on Thu Aug 13 18:48:29 2020

@author: Raul Ortega
"""
import numpy as np
import operator


#============================================================================

# given z, either scalar or array, compute sigmoid for each element
# then return array or scalar
def sigmoid(z):
    return 1/(np.ones((len(z),1)) + np.exp(-z))

#============================================================================



# given some Theta, X, y return the cost (scalar) and gradient (array)
def costFunction(theta, X, y):
    
    # number of training examples
    m = len(y)
    
    # compute cost J (the expression is divided in two terms)
    temp = np.matmul(np.transpose(y), np.log(sigmoid(np.matmul(X, theta))))
    temp1 = np.matmul(np.transpose(np.ones((m,1))-y), np.log(np.ones((m,1))-sigmoid(np.matmul(X, theta))))   
    J = -(temp + temp1)/m
    
    # compute gradient
    grad = np.transpose(np.matmul(np.transpose(sigmoid(np.matmul(X,theta)) - y), X))/m    
    return J, grad


#============================================================================


# given X, y, theta performs gradient descent with learning rate alpha
# a num_iters number of iterations, storing the values of the cost function
# at each iteration and returning J_history and the best theta found
def gradientDescent(X, y, theta, alpha, num_iters):
    
    # initialize J_history array
    J_history = np.zeros((num_iters,1))
    
    # compute cost and gradient given the initial theta values
    temp = costFunction(theta, X, y)
    J = temp[0]
    grad = temp[1]
        
    for iter in range(num_iters):
        # perform a single step of gradient descent
        theta = theta - alpha*grad
        
        # Save the cost J in every iteration and update grad to next step
        temp = costFunction(theta, X, y)
        J = temp[0]
        grad = temp[1]
        J_history[iter] = J
        
        
    return theta, J_history

#=============================================================================

# given an array of strings and a target_word matching_indeces returns an
# array of same length with 1's in the place of every match and 0's elsewhere
def matches_array(words_array, target):
    m = len(words_array)
    temp = np.zeros((m,1))
    for i in range(m):
        if words_array[i][0] == target:
            
            temp[i] = 1
    
    return temp

# ============================================================================

# given a x^(i) and theta_setosa, theta_versicolor, theta_virginica
# for each flower type compute the probability of x^(i) been each type
# of flower and return type of flower with highest probability and that prob.
def predict(x, theta_setosa, theta_versicolor, theta_virginica):
    p_dict = {'Iris-setosa': sigmoid(np.matmul(x, theta_setosa)), 'Iris-versicolor': sigmoid(np.matmul(x, theta_versicolor)), 'Iris-virginica': sigmoid(np.matmul(x, theta_virginica))}
    
    # get the key of the highest value
    hypothesis = max(p_dict.items(), key=operator.itemgetter(1))[0]
    
    # get the highest value of the dict
    p = max(p_dict.values())
    return hypothesis, p
    
    
    
    
    
    
    
    
    
    
    
    