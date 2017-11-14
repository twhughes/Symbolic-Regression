import numpy as np
from generate_examples import *
from equation_tree import *
from encoder import *
import matplotlib.pylab as plt
#from RNN import *
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
N_training = 5

input_features, input_vectors, input_trees, losses = generate_training_examples(N_training,
                                    allowed,tree_depth=tree_depth,const_range=const_range,x_range=x_range,N_points=N_points,
                                    layer_sizes=layer_sizes,activations=activations,
                                    N_epochs=N_epochs,learning_rate=learning_rate,verbose=False)



displ_index = 0

print("String Representation  :   %s" % input_trees[displ_index].string_rep)
print("Tree Representation    :     ")
input_trees[displ_index].print_expanded_tree()
print("One Hot Input Vectors  :   \n %s" % input_vectors[displ_index])
print("Encoded Feature Vector :   \n %s" % input_features[displ_index])
