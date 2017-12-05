import numpy as np
from numpy.random import randint, random
from equation_tree import *
from NN import *
from encoder import *
import sys

def generate_random_tree(allowed, depth=3):
    # generates a random equation tree to some depth

    # separate allowed dictionary into separate classes
    constants = [(name,eq_class,anon_fn) for (name,eq_class,anon_fn) in allowed if eq_class == 'const']
    functions = [(name,eq_class,anon_fn) for (name,eq_class,anon_fn) in allowed if eq_class == 'fn']
    operators = [(name,eq_class,anon_fn) for (name,eq_class,anon_fn) in allowed if eq_class == 'op']

    def get_constant():
        # returns a random constant
        return constants[randint(len(constants))]

    def get_obj():
        # returns a random equation element
        grouped = [constants, functions, operators]
        # first samples randomly from constants, functions, and operators
        group_pick = grouped[randint(3)]
        # then samples randomly from this class
        # this makes it so that adding a lot of one class does not bias the equations to have more elements from that class
        return group_pick[randint(len(group_pick))]

    # recursive function to add a layer to the equation tree based on the class
    def add_layer(node,depth):
        # if we're at the max depth, it must be a constant, because they have no children
        if depth == 0:
            (name,eq_class,anon_fn) = get_constant()
            eq_class = 'const'
        # otherwise, it can be anything
        else:
            name,eq_class,anon_fn = get_obj()
        node.name = name
        node.eq_class = eq_class
        # if it's a function, add one child recursively
        if eq_class == 'fn':
            node.anon_fn = anon_fn
            node.nextR = Node()
            add_layer(node.nextR,depth-1)
        # if it's an operator, add two children recursively            
        elif eq_class == 'op':
            node.anon_fn = anon_fn            
            node.nextL = Node()
            node.nextR = Node()            
            add_layer(node.nextL,depth-1)
            add_layer(node.nextR,depth-1)

    # get new equation tree object and recursively add layers to tree
    tree = EquationTree()
    add_layer(tree.head, depth)
    return tree

def generate_example_list(tree,const_range,x_range,N_points):
    # generates x y sampling points from an equation tree
    # fixes the constants randomly beween range
    tree.fix_constants(const_range)
    N_points = 100
    # randomly sample x and evaluate tree for y points
    x_list = x_range[0] + random(N_points)*(x_range[1]-x_range[0])
    y_list = []
    for i in range(N_points):
        y_list.append(tree.evaluate(x_list[i]))
    return x_list, y_list

def create_index_map(allowed):
    # give a unique index to each equation element, add parentheses too
    index_map = {allowed[i][0]:i+3 for i in range(len(allowed))}
    index_map['('] = 1
    index_map[')'] = 2
    return index_map

def operator_list_to_one_hot(operator_list,index_map):
    # with a list of operators, generate one hot numpy array    
    N = len(index_map)
    M = len(operator_list)
    one_hot = np.zeros((N,M))    
    for i in range(M):
        name = operator_list[i]
        index = index_map[name]
        one_hot[index-1,i] = 1
    return one_hot

def generate_training_examples(N_training,allowed,tree_depth=6,const_range=[-5,5],x_range=[-5,5],N_points=100,
                               layer_sizes=[1,10,10,1],activations=['tanh','tanh','sigmoid'], N_epochs=100,learning_rate=1,verbose=False,uniquify=True,lambda_reg=0.0):  
    # function called by main to get all training examples at once 
    # set up important elements                                
    input_features = []
    input_trees = []
    input_vectors = []
    losses = []
    index_map = create_index_map(allowed)
    seen_equations = set()
    print("")
    # loop through number of training examples desired                                
    for train_i in range(N_training):
        # generate a random equation and get tree
        tree = generate_random_tree(allowed, depth=tree_depth)
        # get the corresponding operator name list     
        operator_list = tree.flatten()
        # convert this to one hot representation   
        eq_string = ''.join(operator_list)
        sys.stdout.write("\rgenerating tree for equation: %s " % (eq_string))
        sys.stdout.flush()            
        one_hot = operator_list_to_one_hot(operator_list,index_map)
        # generate x,y points from the equation
        x_list, y_list = generate_example_list(tree,const_range,x_range,N_points)
        # fit the NN to the x,y points and get the feature vector and loss        
        phi,loss = feature_fit(x_list,y_list,layer_sizes,activations,N_epochs=N_epochs,learning_rate=learning_rate,lambda_reg=lambda_reg)
        # only add if equation is unique
        if uniquify:
            if tree.string_rep not in seen_equations:
                seen_equations.add(tree.string_rep)
                if verbose:
                    print(train_i,loss)
                input_vectors.append(one_hot)
                input_trees.append(tree)            
                losses.append(loss)
                input_features.append(phi)
        else:
            seen_equations.add(tree.string_rep)
            if verbose:
                print(train_i,loss)
            input_vectors.append(one_hot)
            input_trees.append(tree)            
            losses.append(loss)
            input_features.append(phi)            
    return input_features, input_vectors, input_trees, losses


def process_IO_for_keras(input_features,input_vectors,N_training,allowed):
    # Ignore, I was using this to get data into keras but it is no longer needed.
    L_feature = len(input_features[0])
    L_max_eq = max([a.shape[1] for a in input_vectors])
    L_eq_identifier = len(allowed)+2

    resized_input_vectors = []
    for a in input_vectors:
        num_needed = L_max_eq-a.shape[1]
        if num_needed > 0:
            new_array = np.zeros((L_eq_identifier,num_needed))
            resized_input_vectors.append((np.concatenate((a,new_array),axis=1)))
        else:
            resized_input_vectors.append(a)

    inputs = []
    for i in range(N_training):
        phi = np.reshape(np.array(input_features[i]),(L_feature,1))
        #phi = np.zeros((L_feature,1))
        one_hots = resized_input_vectors[i]
        extra_needed = L_feature-L_eq_identifier
        padding = np.zeros((extra_needed,one_hots.shape[1]))
        one_hots = np.concatenate((one_hots,padding),axis=0) 
        inputs.append(np.concatenate((phi,one_hots),axis=1).T)
    #inputs is a list of numpy arrays, [input dimension x number in sequence]
    input_x = L_max_eq+1       # input size dimension
    input_y = L_feature    # number in sequence

    outputs = []
    for i in range(N_training):
        input_array = inputs[i]
        output_array = np.zeros((input_x,input_y))
        output_array[:-1,:] = input_array[1:,:]
        outputs.append(output_array)

    return np.array(inputs),np.array(outputs)








