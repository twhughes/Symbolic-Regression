from NN import NN
import numpy as np

def normalize_ys(y_list):
    min_y = min(y_list)
    y_list = [y-min_y for y in y_list]
    ampl = max(y_list, key=abs)
    y_norm = [y/ampl for y in y_list]
    return y_norm, ampl, min_y


def feature_fit(x_list,y_list,layer_sizes, activations, N_epochs=10, learning_rate = 1.0, threshold = 1e-3):
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
