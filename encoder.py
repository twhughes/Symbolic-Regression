from NN import NN
import numpy as np

EPSILON = 1e-3

def normalize_ys(y_list):
    # normalize all of the y values between 0 and 1 (because this is how the NN likes it with the sigmoid)
    # remember the aplitude and offset, add these to the feature vector
    min_y = min(y_list)
    y_list = [y-min_y for y in y_list]
    ampl = max(y_list, key=abs)
    y_norm = []    
    for y in y_list:
        # NOTE: for constants, ampl = 0 and y = 0 after subtracting min.
        # Therefore, need some way to deal with 0/0 terms
        # set so that if they are less than epsilon, just take as 0/0 = 1
        if abs(y) < EPSILON and ampl < EPSILON:
            y_norm.append(1)
        else:
            y_norm.append(y/ampl)
    return y_norm, ampl, min_y

def feature_fit(x_list,y_list,layer_sizes, activations, N_epochs=10, learning_rate = 1.0, threshold = 1e-3):
    # given x,y pairs, construct the neural network, fit the data, and return the feature vector, including the normalization parameters    
    N = len(x_list)
    y_norm, ampl, min_y = normalize_ys(y_list)
    y_norm = np.array(y_norm)
    neural_net = NN(layer_sizes, activations)
    #neural_net.derivative_check(m=5, epsilon=1e-4, verbose=False)
    inputs = np.reshape(np.array(x_list),(len(x_list),1)).T
    Js = []
    for i in range(N_epochs):
        A = neural_net.forward_prop(inputs)
        dAdZ = A*(1-A)
        J = 1/2.0*np.sum(np.power(A-y_norm,2))/N
        Js.append(J)
        dJdZ = -(A-y_norm)*dAdZ
        neural_net.back_prop(dJdZ)
        neural_net.update_weights(learning_rate=learning_rate)
        if J < threshold:
            break
    feature_vec = neural_net.flatten_parameters()
    feature_vec.append(ampl)
    feature_vec.append(min_y)
    return feature_vec, J
