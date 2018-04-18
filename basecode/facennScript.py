'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your Code Here
    pad = np.ones(shape=(len(training_data),1))
    # print(np.shape(training_data))
    training_data = np.concatenate((training_data, pad), axis=1)
    # print(np.shape(training_data))
    summ1 = training_data.dot(w1.T)
    output_1 = sigmoid(summ1) # Hidden layer Output
    # print(np.shape(output_1))

    sig_pad = np.ones(shape=(len(output_1),1))
    output_1 = np.concatenate((output_1, sig_pad), axis=1)
    summ_2 = output_1.dot(w2.T)
    output_2 = sigmoid(summ_2)
    # print(np.shape(output_2))

    outputclass = np.zeros(np.shape(output_2))
    # print(np.shape(outputclass))
    # print(np.shape(training_label))
    for i in range(len(outputclass)):
        for j in range(np.shape(outputclass)[1]):
            if j == int(training_label[i]):
                outputclass[i][j] = 1

    #-------------------------------------------------------
    # Error function calculation

    first_term = outputclass * np.log(output_2)
    second_term = (1 - outputclass) * np.log(1 - output_2)
    third_term = first_term + second_term

    obj_val = (-1/len(training_data)) * np.sum(third_term)

    #-------------------------------------------------------
    # Regularization
    constant = lambdaval / (2*len(training_data))
    first_term = np.sum(np.square(w1), axis=1)
    second_term = np.sum(np.square(w2), axis=1)
    final_term = np.sum(first_term, axis=0)+np.sum(second_term, axis=0)
    obj_val = obj_val + (constant * final_term)

    #-------------------------------------------------------
    # Calculate Gradient

    gradient_w2 = np.zeros(w2.shape)

    gradient_w1 = np.zeros(w1.shape)
    delta = np.subtract(output_2, outputclass)

    gradient_w2 = (1/len(training_data)) * (delta.T).dot(output_1)
    # print(gradient_w2.shape)
    gradient_w2 = gradient_w2 + ((lambdaval*w2)/len(training_data))

    mult = (1 - output_1[:,:n_hidden])*output_1[:,:n_hidden]
    delta = delta.dot(w2[:,:n_hidden])
    mult = mult * delta

    gradient_w1 = (1/len(training_data))*((mult.T).dot(training_data))
    gradient_w1 = gradient_w1 + (lambdaval*w1/len(training_data))

    obj_grad = np.concatenate((gradient_w1.flatten(), gradient_w2.flatten()),0)

    return (obj_val, obj_grad)
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    # Your code here

    data = np.concatenate((data, np.ones(shape=(len(data), 1))), axis=1)
    summ = data.dot(w1.T)
    output = sigmoid(summ)
    output = np.concatenate((output, np.ones(shape=(len(output), 1))), axis=1)
    summ2 = output.dot(w2.T)
    output_2 = sigmoid(summ2)

    labels = np.argmax(output_2, axis=1)

    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

print("Starting to train...")
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
print("Training complete...")
now = time.time()
#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
print("Time to Train data", now-time.time())

predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
print("Time on Validation data", now-time.time())

predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
print("Time on testing data", now-time.time())
