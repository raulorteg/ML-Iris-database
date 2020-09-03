
"""
Created on Tue Aug 18 07:26:45 2020

@author: Raul Ortega Ochoa
"""

"""
k-means algorithm:
    Take {x^(1), ..., x^(m)}, and K (no. of clusters) as input
    Randomly initialize K cluster centroids mu_1, ..., mu_k
    repeat {
        Cluster assignment (for every training sample x^{i} make c^(i)
        be the label of the closest centroid)
        
        Move centroids (take all the samples assigned to each centroid,
        compute the mean and that will be the next centroid)
        }
"""

import numpy as np
import pandas
import random
from matplotlib import pyplot as plt
from helpers import assignCentroid



"""
    Loading the data
"""

# # import data info details 
# g=open("data/iris.names", "r")
# data_instructions = g.read()
# print(data_instructions)

# loading data into "contents" using pandas
contents = pandas.read_csv('data/iris.data', sep =',', delimiter=',')

# shuffle data
contents = contents.sample(frac=1).reset_index(drop=True) # shuffle rows

contents.columns = ['slength', 'swidth', 'plength', 'pwidth', 'class']
# print(contents)


X = contents[['slength', 'swidth', 'plength', 'pwidth']].to_numpy()
m = len(X) # number of samples
n = len(X[0]) # dimension of samples


"""
    K-means algorithm: initialize K centroids to be K of the samples making
    sure they are not repeated. Then for a number of iterations num_iters
    given repeat:
        {
            1. assign each sample to its closest centroid
            
            2. move each centroid to the average position of the samples
            that have been assigned to it
            }
"""

# set  max initial number of centroids
K_max = 25

if K_max > m:
    print('Error: Cant have more centroids than samples, set K <= m.')

# initialize array where the values of final J for each K will be stored
final_J = []

for K in range(1, K_max + 1):
    # initialize the centroids positions to be those of K random samples
    j = 0
    temp = []
    mu = np.zeros((K,n))
    for i in range(K):
        temp1 = random.randint(0, m-1)
        
        # check centroids wont be repeated
        while temp1 in temp:
            temp1 = random.randint(0, m-1)
        
        temp.append(temp1)
        mu[j][:] = X[temp1][:]
        j = j + 1
    
    # set number of iterations
    num_iters = 100
    
    # initialize J_history array, where J for every iteration is stored
    J_history = []
    
    for num_iter in range(num_iters + 1):
        
        # initialize variables that need to be erased and calculated again
        # after each iteration
        c_vector = []
        J_accum = 0
        last_J = 0
        
        # cluster assignment step 
        for j in range(m):
            temp = assignCentroid(X[j][:], mu)
            
            # cluster assignment
            c_vector.append(temp[0])
            
            # update J accumulator
            J_accum = J_accum + temp[1]
        
        J = J_accum/m
        J_history.append(J)
        
        # Moving centroids step
        # get all x assigned to each cluster together
        for k in range(K):
            temp = [index for index, value in enumerate(c_vector) if value == k]
            foo = np.zeros((1, n))
            for u in temp:
                foo = foo + X[u][:]
            
            # compute new mu_k
            mu[k][:] = foo/len(temp)
    
    final_J.append(J)
    print(f'Finished with J={round(J, 4)} ({K} centroids, {num_iters} iterations) \n')
    
    # # plot cost function vs the number of iterations in the same figure
    # plt.plot(range(num_iters + 1), J_history, 'b')
    # plt.xlabel('no. Iterations')
    # plt.ylabel('Cost Function J')
    # plt.show()
    
"""
    visualize the cost function vs number of centroids. Looking at the elbow
    of the elbow-shaped graph I know the best number of centroids K for the
    data (3 as there are 3 types of flowers in iris.data)
"""

# plot cost function vs number of centroids
plt.plot(range(1, K_max+1), final_J, 'b')
plt.xlabel('no. Centroids')
plt.ylabel('Cost Function J')
plt.show()            
