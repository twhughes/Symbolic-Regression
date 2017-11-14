import numpy as np
from numpy.random import randint, random
from equation_tree import *
from NN import *
from encoder import *
def generate_random_tree(allowed, depth=3):
    # generates a random equation tree to some depth
    constants = [(name,eq_class,anon_fn) for (name,eq_class,anon_fn) in allowed if eq_class == 'const']
    functions = [(name,eq_class,anon_fn) for (name,eq_class,anon_fn) in allowed if eq_class == 'fn']
    operators = [(name,eq_class,anon_fn) for (name,eq_class,anon_fn) in allowed if eq_class == 'op']

    def get_constant():
        return constants[randint(len(constants))]
    def get_obj():
        grouped = [constants, functions, operators]
        group_pick = grouped[randint(3)]
        return group_pick[randint(len(group_pick))]
    def add_layer(node,depth):
        if depth == 0:
            (name,eq_class,anon_fn) = get_constant()
            eq_class = 'const'
        else:
            name,eq_class,anon_fn = get_obj()
        node.name = name
        node.eq_class = eq_class
        if eq_class == 'fn':
            node.anon_fn = anon_fn
            node.nextR = Node()
            add_layer(node.nextR,depth-1)
        elif eq_class == 'op':
            node.anon_fn = anon_fn            
            node.nextL = Node()
            node.nextR = Node()            
            add_layer(node.nextL,depth-1)
            add_layer(node.nextR,depth-1)

    tree = EquationTree()
    add_layer(tree.head, depth)
    return tree

def generate_example_list(tree,const_range,x_range,N_points):
    tree.fix_constants(const_range)
    N_points = 100
    x_list = x_range[0]+(x_range[1]-x_range[0])*random(N_points)
    y_list = []
    for i in range(N_points):
        y_list.append(tree.evaluate(x_list[i]))
    return x_list, y_list

def create_index_map(allowed):
    index_map = {allowed[i][0]:i+3 for i in range(len(allowed))}
    index_map['('] = 1
    index_map[')'] = 2
    return index_map

def operator_list_to_one_hot(operator_list,index_map):
    N = len(index_map)
    M = len(operator_list)
    one_hot = np.zeros((N,M))    
    for i in range(M):
        name = operator_list[i]
        index = index_map[name]
        one_hot[index-1,i] = 1
    return one_hot

def generate_training_examples(N_training,allowed,tree_depth=6,const_range=[-5,5],x_range=[-5,5],N_points=100,
                               layer_sizes=[1,10,10,1],activations=['tanh','tanh','sigmoid'], N_epochs=100,learning_rate=1,verbose=False):    
    input_features = []
    input_trees = []
    input_vectors = []
    losses = []
    index_map = create_index_map(allowed)
    for train_i in range(N_training):
        tree = generate_random_tree(allowed, depth=tree_depth)
        operator_list = tree.flatten()
        one_hot = operator_list_to_one_hot(operator_list,index_map)
        input_vectors.append(one_hot)
        input_trees.append(tree)
        x_list, y_list = generate_example_list(tree,const_range,x_range,N_points)
        phi,loss = feature_fit(x_list,y_list,layer_sizes,activations,N_epochs=N_epochs,learning_rate=learning_rate)
        if verbose:
            print(train_i,loss)
        losses.append(loss)
        input_features.append(phi)

    return input_features, input_vectors, input_trees, losses


