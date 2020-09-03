"""
Created on Sat Aug 22 13:25:06 2020

@author: Raul Ortega Ochoa
"""

import numpy as np
import pandas
from matplotlib import pyplot as plt
from helpers import sigmoid, derSigmoid, reshape_class, predict_Iris
from datetime import datetime

# start time counter
startTime = datetime.now()


"""
    Load the data, shuffle it and divide the train (70%) and test (30%) set.
    Then load train set into X_train, y_train and test set into X_test, 
    y_test as numpy arrays.
"""

# # load the data info
# with open('data/iris.names', 'r') as g:
#     data_info = g.read()
#     print(data_info)


# load the data from iris.data with pandas and add 1's column
data = pandas.read_csv('data/iris.data', sep=',', delimiter=',')
# data.insert(0, 'ones',  np.ones(len(data)), True)
data.columns = ['slength', 'swidth', 'plength', 'pwidth', 'class']


# mean normalization on the data
data1 = data[['slength', 'swidth', 'plength', 'pwidth']] # select rows
normalized_data=(data1-data1.mean())/data1.std() # normalize selected rows
normalized_data.insert(4, 'class', data[['class']], True) # add class row


# shuffle and divide into train-test (70-30%) and X-Y matrixes
normalized_data = normalized_data.sample(frac=1).reset_index(drop=True) # shuffle rows

m = len(normalized_data) # number of samples
limiter = round(0.7*m)

train_set = normalized_data[0:limiter-1] # take 70% as train set
test_set = normalized_data[limiter:m-1] # take rest (30%) as test set

X_train = train_set[['slength', 'swidth', 'plength', 'pwidth']].to_numpy()
y_train = train_set[['class']].to_numpy()
X_test = test_set[['slength', 'swidth', 'plength', 'pwidth']].to_numpy()
y_test = test_set[['class']].to_numpy()
   

"""
    Choosing the NN arquitecture: 4 input units, two 6-unit hidden layers
    and a 3-unit output layer.
"""

# choose arquitecture setting the dimensions of the layers
dim_layer1 = 4
dim_layer2 = 6
dim_layer3 = 6
dim_layer4 = 3

# set number of iterations
num_iters = 1000

# set learning rate
alpha = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

# set value of reg_lambda
reg_lambda = [0, 0.1, 0.5, 1, 3, 5, 7, 10]

# some useful data of the dimensions of dataset
m_train = len(X_train)
m_test = len(X_test)

test_accuracy_array = []
train_accuracy_array = []

for m in range(8):
    for h in range(8):
        # initialize weight's matrices randomly to break the symmetry
        Theta_1 = (np.sqrt(6/(dim_layer1+dim_layer2)))*np.random.rand(dim_layer2, dim_layer1) # maps first layer to second 
        Theta_2 = (np.sqrt(6/(dim_layer2+dim_layer3)))*np.random.rand(dim_layer3, dim_layer2) # maps second layer to third
        Theta_3 = (np.sqrt(6/(dim_layer3+dim_layer4)))*np.random.rand(dim_layer4, dim_layer3) # maps third layer to fourth
        
        # initialize the accumulators Delta_l with l =1,2,3,4
        Delta_1 = np.zeros((dim_layer2, dim_layer1))
        Delta_2 = np.zeros((dim_layer3, dim_layer2)) 
        Delta_3 = np.zeros((dim_layer4, dim_layer3))
        
        
        for v in range(num_iters):
            
            # initialize J/restart J = 0
            J = 0
            
            # the following is done for every sample of the train set: 
            # 1. compute the predictions for one sample, 2. compare them with what was
            # expected, then backpropagate errors backwards through the NN
            # changing Theta_j. Return to step one for the next sample.
            for w in range(m_train):
                """    
                    %%%%%% Forward propagation %%%%%%
                    Using sigmoid (logistic) activation function
                    a_j[i] : activation of unit i in layer j
                    Theta_j : Weights matrix controlling the mapping of j_th layer to j+1
                    Theta_j has dim s_(j+1) x (s_j) 
                """
                
                a_1 = np.transpose(X_train[w]) # activation values for first layer
                a_1 = np.reshape(a_1, (len(a_1), 1))
                
                a_2 = sigmoid(np.matmul(Theta_1, a_1))
                a_2 = np.reshape(a_2, (len(a_2), 1))
                
                a_3 = sigmoid(np.matmul(Theta_2, a_2))
                a_3 = np.reshape(a_3, (len(a_3), 1))
                
                a_4 = sigmoid(np.matmul(Theta_3, a_3)) # predition layer
                a_4 = np.reshape(a_4, (len(a_4), 1))
                
                # accumulate cost function
                temp = reshape_class(y_train[w])
                len_temp = len(temp)
                
                # J_part1 = np.multiply(temp, np.log(sigmoid(a_4)))
                # J_part2= np.multiply((np.ones((len_temp, 1))-temp), np.log(np.ones((len(a_4), 1))-sigmoid(a_4)))
                # J = J + sum(J_part1 + J_part2)/(-m_train)
                # J = J + reg_lambda*(sum(sum(np.square(Theta_1))) + sum(sum(np.square(Theta_2))) + sum(sum(np.square(Theta_3))))/(2*m_train)
                
                """
                    %%%%% Backpropagation %%%%
                    L = 4 (no. layers)
                """
                delta_4 = a_4 - reshape_class(y_train[w]) # compare predicted values
                delta_3 = np.multiply(np.matmul(np.transpose(Theta_3), delta_4), derSigmoid(np.matmul(Theta_2, a_2)))
                delta_2 = np.multiply(np.matmul(np.transpose(Theta_2), delta_3), derSigmoid(np.matmul(Theta_1, a_1)))
                
                # update accumulators
                Delta_1 = Delta_1 + np.matmul(delta_2, np.transpose(a_1))
                Delta_2 = Delta_2 + np.matmul(delta_3, np.transpose(a_2))
                Delta_3 = Delta_3 + np.matmul(delta_4, np.transpose(a_3))
            
            
            # # append cost function to J_history
            # J_history.append(J)
            
            # compute D_l (=der(J)/dif(Theta_l))
            D_1 = Delta_1/m_train + reg_lambda[h]*Theta_1 
            D_2 = Delta_2/m_train + reg_lambda[h]*Theta_2
            D_3 = Delta_3/m_train + reg_lambda[h]*Theta_3
            
            # update Thetas weight with a gradient descent step
            Theta_1 = Theta_1 - alpha[m]*D_1
            Theta_2 = Theta_2 - alpha[m]*D_2
            Theta_3 = Theta_3 - alpha[m]*D_3
            
        """
            Visualizing the evolution of the cost function with every iteration:
            plot J_history vs num_iters to adjust learning rate alpha and num_iters
        """
        
        # # plot every cost function vs the number of iterations in the same figure
        # plt.plot(range(1, num_iters + 1), J_history, 'b')
        # plt.xlabel('no. Iterations')
        # plt.ylabel('Cost Function J')
        # plt.show()
        
        
        """
            Now lets see how it manages the unseen test set. For every sample
            of the test set make a prediction and compare with what is known as the 
            correct answer. accuracy = successful predictions / total predictions
        """
        
        # initialize some counter
        correct_pred = 0
        
        for w in range(m_test):
            
            # perform forward propagation
            a_1 = np.transpose(X_test[w]) # activation values for first layer
            a_1 = np.reshape(a_1, (len(a_1), 1))
                
            a_2 = sigmoid(np.matmul(Theta_1, a_1))
            a_2 = np.reshape(a_2, (len(a_2), 1))
                
            a_3 = sigmoid(np.matmul(Theta_2, a_2))
            a_3 = np.reshape(a_3, (len(a_3), 1))
            
            a_4 = sigmoid(np.matmul(Theta_3, a_3)) # predition layer
            a_4 = np.reshape(a_4, (len(a_4), 1))
            
            # feed the predicition a_4 to predict_Iris(a_4) that returns the class
            # with highest chances in the prediction a_4. If correct add one to
            # correct_pred counter
            if y_test[w] == predict_Iris(a_4):
                correct_pred = correct_pred + 1
        
        # compute accuracy (%)
        accuracy_test = (correct_pred/m_test)*100
        test_accuracy_array.append(accuracy_test)
        
        """
            lets see how it manages the training set set. For every sample
            of the training set make a prediction and compare with what is known as the 
            correct answer. accuracy = successful predictions / total predictions
        """
        
        # initialize some counter
        correct_pred = 0
        
        for w in range(m_train):
            
            # perform forward propagation
            a_1 = np.transpose(X_train[w]) # activation values for first layer
            a_1 = np.reshape(a_1, (len(a_1), 1))
                
            a_2 = sigmoid(np.matmul(Theta_1, a_1))
            a_2 = np.reshape(a_2, (len(a_2), 1))
                
            a_3 = sigmoid(np.matmul(Theta_2, a_2))
            a_3 = np.reshape(a_3, (len(a_3), 1))
            
            a_4 = sigmoid(np.matmul(Theta_3, a_3)) # predition layer
            a_4 = np.reshape(a_4, (len(a_4), 1))
            
            # feed the predicition a_4 to predict_Iris(a_4) that returns the class
            # with highest chances in the prediction a_4. If correct add one to
            # correct_pred counter
            if y_train[w] == predict_Iris(a_4):
                correct_pred = correct_pred + 1
        
        
        # compute accuracy (%)
        accuracy_train = (correct_pred/m_train)*100
        train_accuracy_array.append(accuracy_train)
        print(f'train: {round(accuracy_train)}%, test: {round(accuracy_test)}% (alpha: {alpha[m]}, lambda:{reg_lambda[h]})')



# print time it took to execute
finishTime = datetime.now()
diff = (finishTime - startTime).total_seconds()
print(f'finished in {round(diff,3)} s.')
