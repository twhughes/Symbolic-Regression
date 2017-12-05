import numpy as np

# activation function dictionary
# relu and linear don't work as expected, stick to tanh and sigmoid
possible_activations = {
    'tanh'    : lambda z : np.tanh(z),
    'relu'    : lambda z : np.maximum(z,0),
    'sigmoid' : lambda z : 1./(1+np.exp(-z)),
    'linear'  : lambda z : z
}
# activation function derivative dictionary
#     note: defined in terms of the activations (a) for speed up in computation
possible_activation_derivatives = {
    'tanh'    : lambda a : 1-np.power(a,2),
    'relu'    : lambda a : 1.*(a>0),
    'sigmoid' : lambda a : a*(1-a),
    'linear'  : lambda a : 1.0
}

class NN:

    def __init__(self, layer_sizes, activations):
        # initialization method:
        #   INPUTS:
        #     layer_sizes = list of integers [input layer size, hidden layer 1 size, hidden layer 2 size, ... output layer size]
        #     activations = list of strings  [activation 1, activation 2, ... activation N]
        #   COMPUTES:
        #     creates random initial weights and biases
        #     stores activation functions and their derivatives
        #   RETURNS:
        #     neural network class
        assert (len(layer_sizes) == len(activations) + 1), "number of layers must be one greater than number of activations"
        self.weights = {}
        self.biases = {}
        self.layer_sizes = layer_sizes
        self.N_layers = len(layer_sizes)
        self.activations = {}
        self.activation_derivatives = {}
        self.cache = {}
        self.grads = {}
        np.random.seed(1)
        for l in range(self.N_layers-1):
            self.weights['W'+str(l+1)] = 2.*np.random.rand(layer_sizes[l+1],layer_sizes[l])-1.
            self.biases['b'+str(l+1)]  = np.zeros((layer_sizes[l+1],1),dtype=np.float32)
            self.activations['a'+str(l+1)] = possible_activations[activations[l]]
            self.activation_derivatives['a'+str(l+1)] = possible_activation_derivatives[activations[l]]
        np.random.seed(None)

    def forward_prop(self, input_vec):
        # forward propagation method:
        #   INPUTS:
        #     input_vec = np array of size (N_inputs, m) where m is the number of training examples to compute
        #   COMPUTES:
        #     performs forward propagation in the netowrk
        #     saves Z[l] = W[l]a[l-1]+b[l] and activations a[l] in the cache for backpropagation later
        #   RETURNS:
        #     output layer output           
        m = (input_vec.shape)[1]
        assert (input_vec.shape[0]==self.layer_sizes[0])
        a_prev = input_vec        
        self.cache['a0'] = input_vec
        for l in range(1,self.N_layers):
            zl = np.dot(self.weights['W'+str(l)],a_prev) + self.biases['b'+str(l)]
            self.cache['z'+str(l)] = zl
            al = self.activations['a'+str(l)](zl)
            self.cache['a'+str(l)] = al
            a_prev = al
        return al


    def back_prop(self, dZ_end): 
        # backward propagation method:
        #   INPUTS:
        #     dZ_end = np array of size (N_output, m) which is the derivative of the loss function with respect to the last Z[l] output
        #   COMPUTES:
        #     performs backpropagation on the netowrk
        #     saves gradients in the self.grads{} cache
        #   RETURNS:
        #     None           
        m = dZ_end.shape[1]
        assert (dZ_end.shape[0]==self.layer_sizes[self.N_layers-1])        
        dZ = dZ_end
        for l in range(self.N_layers-1,1,-1):
            dWl = 1./m*np.dot(dZ,self.cache['a'+str(l-1)].T)
            dbl = 1./m*np.sum(dZ,axis=1,keepdims=True)     
            self.grads['dW'+str(l)] = dWl
            self.grads['db'+str(l)] = dbl 
            dZ = np.multiply(np.dot(self.weights['W'+str(l)].T, dZ), self.activation_derivatives['a'+str(l-1)](self.cache['a'+str(l-1)]))
        dW1 = 1./m*np.dot(dZ,self.cache['a0'].T)
        db1 = 1./m*np.sum(dZ,axis=1,keepdims=True)     
        self.grads['dW1'] = dW1
        self.grads['db1'] = db1     

    def update_weights(self, learning_rate=0.01, lambda_reg=0):
        # weight update method:
        #   INPUTS:
        #     learning_rate = float, amount to increment weights
        #   COMPUTES:
        #     takes gradient cache and updates each weight and bias according to learning_rate
        #   RETURNS:
        #     None  
        # note: expand to incorporate other learning algorithms later       
        for l in range(1,self.N_layers):
            self.weights['W'+str(l)] = self.weights['W'+str(l)] + learning_rate*self.grads['dW'+str(l)] - lambda_reg*self.weights['W'+str(l)]
            self.biases[ 'b'+str(l)] = self.biases[ 'b'+str(l)] + learning_rate*self.grads['db'+str(l)] - lambda_reg*self.biases[ 'b'+str(l)]

    def derivative_check(self, m=5, epsilon=1e-4, verbose=False):
        # numerical derivative checking method: use to make sure it is working correctly
        #   INPUTS:
        #     m       = integer, number of training examples to simulate
        #     epsilon = float,   amount to perturb weights
        #     verbose = boolean, whether to print out all of the derivatives
        #   COMPUTES:
        #     takes numerical derivative of loss function with respect to each weight and bias
        #     compares numerical calculation with backprop result
        #     gives errors if they are different (this function used mainly for debugging purposes, don't do it every time)
        #   RETURNS:
        #     {True/False} if numerical derivatives {match/do not match} backprop results 
        input_vec  = np.random.rand((self.weights['W1'].shape)[1],m)
        y = np.random.rand((self.weights['W'+str(self.N_layers-1)].shape)[0],m)
        output_vec = self.forward_prop(input_vec)
        self.back_prop(output_vec-y)
        correct = True
        for l in range(1,self.N_layers):
            Wl = self.weights['W'+str(l)]
            nx,ny = Wl.shape
            for i in range(nx):
                for j in range(ny):
                    self.weights['W'+str(l)][i,j] += epsilon
                    out_plus = self.forward_prop(input_vec)
                    loss_plus = -(y*np.log(out_plus)+(1-y)*np.log(1-out_plus))/m
                    self.weights['W'+str(l)][i,j] -= 2*epsilon
                    out_minus = self.forward_prop(input_vec) 
                    loss_minus = -(y*np.log(out_minus)+(1-y)*np.log(1-out_minus))/m
                    self.weights['W'+str(l)][i,j] += epsilon
                    deriv_numerical = np.sum((loss_plus-loss_minus)/2./epsilon)
                    deriv_back_prop = self.grads['dW'+str(l)][i][j]
                    if verbose:
                        print('layer : '+str(l)+' inedeces : ('+str(i)+','+str(j)+')')
                        print('    numerical  : '+str(deriv_numerical))
                        print('    analytical : '+str(deriv_back_prop))
                    if (np.abs(deriv_numerical - deriv_back_prop) > 1e-5):
                        correct = False
        if correct:
            print('success: backpropagation matches numerical derivatives.')
        else:   
            print('WARNING: backpropagation does not match numerical derivatives, please check')                     
        return correct

    def flatten_parameters(self):
        # gets all of the parameters in the network, stick them into a vector and return them (this is our feature vector)
        feature_vec = []
        for l in range(1,self.N_layers):
            W_array = self.weights['W'+str(l)]
            b_array = self.biases[ 'b'+str(l)]            
            Nx,Ny = W_array.shape
            for i in range(Nx):
                for j in range(Ny):
                    feature_vec.append(W_array[i,j])
                feature_vec.append(b_array[i,0])
        return feature_vec





