'''
Accelerating Symbolic Regression with Deep Learning
By Tyler Hughes, Siddharth Buddhiraju and Rituraj
For CS 221, Fall 2017-2018

Provides helper function for fitting the x,y points and 
extracting features.
'''

from NN import NN
import numpy as np

def feature_fit(x_list,y_list,layer_sizes, activations, N_epochs=10, learning_rate = 1.0, threshold = 1e-3, lambda_reg=0.0):
    # given x,y pairs, construct the neural network, fit the data, and return the feature vector, including the normalization parameters    
    N = len(x_list)
    y_norm = np.array(y_list)
    neural_net = NN(layer_sizes, activations)
    #neural_net.derivative_check(m=5, epsilon=1e-4, verbose=False)
    inputs = np.reshape(np.array(x_list),(len(x_list),1)).T
    Js = []
    for i in range(N_epochs):
        A = neural_net.forward_prop(inputs)
        dAdZ = 1
        J = 1/2.0*np.sum(np.power(A-y_norm,2))/N
        Js.append(J)
        dJdZ = -(A-y_norm)*dAdZ
        neural_net.back_prop(dJdZ)
        neural_net.update_weights(learning_rate=learning_rate,lambda_reg=lambda_reg)
        if J < threshold:
            break
    feature_vec = neural_net.flatten_parameters()
    return feature_vec, J