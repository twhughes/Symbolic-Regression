import numpy as np
import csv

from generate_examples_helper import generate_training_examples

# stores the allowable function elements, 
  # first element is the name of the function element
  # second is the class (must be one of {'fn','op','const'} for function, operator, and constant
  # third is lambda expression for computing the element, return True for constants
allowed = [('sin','fn',lambda x: np.sin(x)),
           ('x','const',lambda _:True),
           ('+','op',lambda x,y: x+y),
           ('*','op',lambda x,y: x*y),
           ('tanh','fn',lambda x: np.tanh(x)),
           #('c','const',lambda _: True),
           #('b','const',lambda _: True),
           ('exp','fn',lambda x: np.exp(x)),
           ('log','fn',lambda x: np.log(np.abs(x))),
           ('cosh','fn',lambda x: np.cosh(x)),           
           ('sinh','fn',lambda x: np.sinh(x))]  


# parameters for fitting to the datapoints
const_range = [-5,5]             # when determining constants, sample randomly in this range
x_range = [-5,5]                 # when sampling x points, sample randomly in this range
N_points = 50                    # number of x,y pairs to generate in training examples
N_epochs = 100                   # number of training epochs for neural network fitting of x,y pairs
tree_depth = 1                 # max equation tree depth (depth of 0 is x,c by default, depth of 1 could be sin(x), cos(c), c, etc.)
learning_rate = 1                # neural network learning rate
layer_sizes = [1,4,4,1]     # layer sizes in neural network, first must be 1, last must be 1, middle layers can change size or add more
activations = ['tanh','tanh','sigmoid']  # NN activations, list must be 1 less in length than above layer sizes.  Stick to pattern of tanh, tanh, ... tanh, sigmoid
N_training = 200                # number of training steps for the NN fit.

print('generating %s training examples up to tree depth of %s...'%(N_training,tree_depth))
# this generates random equations, randomly sets constants, generates the x,y pairs, fits them to neural network and returns feature vectors
input_features, input_vectors, input_trees, losses = generate_training_examples(N_training,
                                    allowed,tree_depth=tree_depth,const_range=const_range,x_range=x_range,N_points=N_points,
                                    layer_sizes=layer_sizes,activations=activations,
                                    N_epochs=N_epochs,learning_rate=learning_rate,verbose=False)
# input features: list of feature vectors (flattened weights and biases from NN)
# input vectors: list of list of one_hot vector corresponding to the equations generated.
# input trees: list of tree structures corresponding to the equations
# losses: list of losses for each of the NN fits (to see if the fitting was accurate enough)
# y_list: ignore, y points corresponding to x points generated from sampling.

# write the equation strings and feature vectors to files in the data/ directory
print('    started with %s training examples'%N_training)
N_training = len(input_features)  # need to account for the fact that we removed duplicated training examples
print('    have %s training examples after removing duplicates'%N_training)
print('saving data to files...')
g = open('data/desired_equation_components.txt', 'w')
for i in range(N_training):
    eq_tree = input_trees[i]
    eq_string = eq_tree.flatten()
    g.write(','.join(eq_string)+'\n')  # python will convert \n to os.linesep
g.close()
np.savetxt("data/encoded_states.txt", np.array(input_features), delimiter=",")

