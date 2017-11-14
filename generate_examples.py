import numpy as np
from numpy.random import randint, random
from equation_tree import *


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
            node.nextL = Node()
            add_layer(node.nextL,depth-1)
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