'''
Accelerating Symbolic Regression with Deep Learning
By Tyler Hughes, Siddharth Buddhiraju and Rituraj
For CS 221, Fall 2017-2018

Script to generate datasets to be stored in ./data/
'''

import numpy as np
import csv
import dill    # needed for pickling lambda expressions
import pickle  # comes pre-installed

from generate_examples_helper import generate_training_examples

  # stores the allowable function elements,
  # first element is the name of the function element
  # second is the class (must be one of {'fn','op','const'} for function, operator, and constant
  # third is lambda expression for computing the element, return True for constants
allowed = [('sin','fn',lambda x: np.sin(x)),
           #('cos','fn',lambda x: np.cos(x)),
           ('x','var',lambda _:True),
           ('+','op',lambda x,y: x+y),
           #('*','op',lambda x,y: x*y),
           ('tanh','fn',lambda x: np.tanh(x)),
           ('c','const',lambda _: True)]
           #('b','const',lambda _: True),
           #('log','fn',lambda x: np.log(np.abs(x)))]
           #('cosh','fn',lambda x: np.cosh(x))]

# parameters for fitting to the datapoints
const_range = [-5,5]              # when determining constants, sample randomly in this range
x_range = [-1,1]                  # when sampling x points, sample randomly in this range
N_points = 50                     # number of x,y pairs to generate in training examples
N_epochs = 1000                   # number of training epochs for neural network fitting of x,y pairs
tree_depth = 4                    # max equation tree depth (depth of 0 is x,c by default, depth of 1 could be sin(x), cos(c), c, etc.)
learning_rate = .1                # neural network learning rate
layer_sizes = [1,4,1]             # layer sizes in neural network, first must be 1, last must be 1, middle layers can change size or add more
activations = ['tanh','linear']   # NN activations, list must be 1 less in length than above layer sizes.  Stick to pattern of tanh, tanh, ... tanh, sigmoid
N_training = 1500                   # number of training steps for the NN fit.

print('generating %s training examples up to tree depth of %s...'%(N_training,tree_depth))
# this generates random equations, randomly sets constants, generates the x,y pairs, fits them to neural network and returns feature vectors
input_features, input_vectors, input_trees, losses = generate_training_examples(N_training,
                                    allowed,tree_depth=tree_depth,const_range=const_range,x_range=x_range,N_points=N_points,
                                    layer_sizes=layer_sizes,activations=activations,
                                    N_epochs=N_epochs,learning_rate=learning_rate,verbose=False,uniquify=False,lambda_reg=.01)
# input features: list of feature vectors (flattened weights and biases from NN)
# input vectors: list of list of one_hot vector corresponding to the equations generated.
# input trees: list of tree structures corresponding to the equations
# losses: list of losses for each of the NN fits (to see if the fitting was accurate enough)
# y_list: ignore, y points corresponding to x points generated from sampling.

# write the equation strings and feature vectors to files in the data/ directory
print('    started with %s training examples'%N_training)
N_training = len(input_features)  # need to account for the fact that we removed duplicated training examples
print('    have %s training examples if removing duplicates'%N_training)
print('saving data to files...')
g = open('data/desired_equation_components_d'+str(tree_depth)+'.txt', 'w')
for i in range(N_training):
    eq_tree = input_trees[i]
    eq_string = eq_tree.flatten()
    g.write(','.join(eq_string)+'\n')  # python will convert \n to os.linesep
g.close()
np.savetxt("data/encoded_states_d"+str(tree_depth)+".txt", np.array(input_features), delimiter=",")
pickle.dump(input_trees, open( "data/equation_trees_d"+str(tree_depth)+".p", "wb" ) )
pickle.dump(allowed, open( "data/allowed_d"+str(tree_depth)+".p", "wb" ) )
