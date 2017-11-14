import numpy as np
from generate_examples import *
from equation_tree import *
from encoder import *
import matplotlib.pylab as plt
from RNN import *
from NN import *

#allowed = [('sin','fn',lambda x: np.sin(x)),('x','const',lambda _:True),('+','op',lambda x,y: x+y),('cos','fn',lambda x: np.cos(x)),('c','const',lambda _:True)]
allowed = [('sin','fn',lambda x: np.sin(x)),
           ('x','const',lambda _:True),
           ('+','op',lambda x,y: x+y),
           ('cos','fn',lambda x: np.cos(x))]           

const_range = [-5,5]
x_range = [-5,5]
N_points = 50
N_epochs = 100
tree_depth = 6
learning_rate = 1
layer_sizes = [1,10,10,1]
activations = ['tanh','tanh','sigmoid']
N_training = 40
input_features = []
input_trees = []
losses = []
for train_i in range(N_training):
    print(train_i)
    tree = generate_random_tree(allowed, depth=tree_depth)
    input_trees.append(tree)
    x_list, y_list = generate_example_list(tree,const_range,x_range,N_points)
    phi,loss = feature_fit(x_list,y_list,layer_sizes, activations,N_epochs=N_epochs,learning_rate=learning_rate)
    losses.append(loss)
    input_features.append(phi)

#print(input_features[4])
#input_trees[4].print_tree()

build_LSTM_model(allowed, input_features, input_trees)