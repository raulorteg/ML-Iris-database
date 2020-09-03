"""
Created on Wed Aug 12 19:14:52 2020

@author: Raul Ortega Ochoa

1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica

8. Missing Attribute Values: None
"""
import numpy as np
import pandas
from matplotlib import pyplot as plt

from helpers import *

# # import data info details 
# g=open("iris.names", "r")
# data_instructions = g.read()
# print(data_instructions)

# load data into "contents" using pandas
contents = pandas.read_csv('iris.data', sep=',', delimiter=',')
contents.columns = ['slength','swidth', 'plength', 'pwidth', 'class']
# add a column of ones
contents.insert(0, "ones", np.ones(len(contents)), True) 
print(contents)

# shuffle data then divide train test set
temp = contents.sample(frac=1).reset_index(drop=True) # shuffle rows

set_limiter = round(0.7*len(temp))
contents_train = temp[0:set_limiter] # take 70% as train set
contents_test = temp[set_limiter:len(temp)] # take rest for test set



X_train = contents_train[['ones', 'slength','swidth', 'plength', 'pwidth']].to_numpy()
y_train = contents_train[['class']].to_numpy()

X_test = contents_test[['ones','slength','swidth', 'plength', 'pwidth']].to_numpy()
y_test = contents_test[['class']].to_numpy()



# some useful parameters for the dimensions
m_train = len(contents_train)
m_test = len(contents_test)
n = len(X_train[0])


"""
    now I have the dataset loaded and divided into X, y matrices
    and training/test set (they are arrays). Note: Every time the script 
    is executed the shuffle will be diferent.
    
    next I will use logisitc regression with One-vs-all strategy to 
    distinguish between flowers. I have 3 different types of flowers so
    I need to train 3 different classifiers
"""

# create the 3 arrays y_virginica, y_setosa, y_versicolor such that e.g for 
# y_virginica if y(j) = 'Iris_virginica' then y_virginica(j) = 1, and 
# y_virginica(j) = 0 otherwise

y_virginica =  matches_array(y_train, 'Iris-virginica')
y_setosa =  matches_array(y_train, 'Iris-setosa')
y_versicolor =  matches_array(y_train, 'Iris-versicolor')

# initialize fitting parameters
initial_theta = np.zeros((n, 1));

# set learning rate
alpha = 0.1

# set number of iterations
num_iters = 5000


# run gradient descent for every type of flower
temp = gradientDescent(X_train, y_virginica, initial_theta, alpha, num_iters)
theta_virginica = temp[0]
J_history_virginica = temp[1]

temp = gradientDescent(X_train, y_setosa, initial_theta, alpha, num_iters)
theta_setosa = temp[0]
J_history_setosa = temp[1]

temp = gradientDescent(X_train, y_versicolor, initial_theta, alpha, num_iters)
theta_versicolor = temp[0]
J_history_versicolor = temp[1]

"""
    now I have the best theta for every type of flower and their J_history
    for every iteration. Now I plot J_history vs iter_number to visualize if
    J is decreasing with time
"""

# plot every cost function vs the number of iterations in the same figure
plt.plot(range(1, num_iters + 1), J_history_setosa, 'b')
plt.plot(range(1, num_iters + 1), J_history_versicolor, 'r')
plt.plot(range(1, num_iters + 1), J_history_virginica, 'g')
plt.legend(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'))
plt.xlabel('no. Iterations')
plt.ylabel('Cost Function J')
plt.show()


"""
    now that best thetas are known for each flower type I make the predictions
    given a set (x^(i)) to know the probability of x^(i) corresponding to each
    flower type and select the type with the highest probability. 
"""
i = 0
# counter of correct predictions
matches_in_test = 0

for x in X_test:
    # make predictions
    temp = predict(x, theta_setosa, theta_versicolor, theta_virginica)
    hypothesis = temp[0]
    highest_p = temp[1]
    print(y_test[i], hypothesis, highest_p)
    
    # update counter if correct prediction
    if y_test[i] == hypothesis:
        matches_in_test = matches_in_test + 1
    i = i + 1
# compute accuracy (%) for test set and display it
accuracy = (matches_in_test/m_test)*100
print(f'accuracy: {accuracy} %')
print()

"""
    Finally, given a new x_new array return the type of flower it has the
    higher chances to belong to and its probability
"""

x_new = np.array([1, 4.5, 3.5, 3, 1])
print(predict(x_new, theta_setosa, theta_versicolor, theta_virginica))


