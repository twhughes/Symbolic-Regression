import numpy as np
from generate_examples import *
from equation_tree import *
from encoder import *
import matplotlib.pylab as plt
from RNN import *
from NN import *
"""
if __name__ == '__main__':
    enc,dec,errors = train_RNNs()
    examples = open('./data/test/*')
    avg_error = 0.0
    for eq in examples:
        phi = generate_feature_vector(eq,N_points)
        eq_guess, error = dec(phi)
        avg_error += error/len(examples)
    errors['test'] = avg_error

    print('train error : '+str(errors['train']))
    print('dev error   : '+str(errors['dev']))
    print('test error  : '+str(errors['test']))

if __name__ == '__main__':
    eq = input('please enter equation')
    phi = generate_feature_vector(eq,N_points)
    eq_guess, error = dec(phi)
    print('we guessed your equation to be '+str(eq_guess)+' with an error of: '+str(error))
"""



allowed = [('sin','fn',lambda x: np.sin(x)),('x','const',lambda _:True),('+','op',lambda x,y: x+y),('cos','fn',lambda x: np.cos(x)),('c','const',lambda _:True)]
allowed = [('sin','fn',lambda x: np.sin(x)),
           ('x','const',lambda _:True),
           ('+','op',lambda x,y: x+y),
           ('cos','fn',lambda x: np.cos(x))]           

const_range = [-5,5]
x_range = [-5,5]
N_points = 100
N_epochs = 200
tree_depth = 6
learning_rate = 1
layer_sizes = [1,10,10,1]
activations = ['tanh','tanh','sigmoid']
N_training = 100
input_features = []
input_trees = []
for train_i in range(N_training):
    print(train_i)
    tree = generate_random_tree(allowed, depth=tree_depth)
    input_trees.append(tree)
    x_list, y_list = generate_example_list(tree,const_range,x_range,N_points)
    phi = feature_fit(x_list,y_list,layer_sizes, activations,N_epochs=N_epochs,learning_rate=learning_rate)
    input_features.append(phi)

print(input_features[4])
input_trees[4].print_tree()