from numpy.random import random

# implements an equation tree class
class Node:
    def __init__(self):
        self.val = None        # value of node, corresponding to the value up to that node when x is entered and constants are specified
        self.name = None       # name of node
        self.eq_class = None   # class of node
        self.anon_fn = None    # anonomous function of node
        self.nextL = None      # left child (specified for operator only, None for constant and function)
        self.nextR = None      # right child (specified for operator and function, None for constant)
        self.string_rep = ''   # string represntation (unused)
        
class EquationTree:

    def __init__(self):
        # create head node on initialization
        self.head = Node()

    def __str__(self):
        # print(tree) just gives empty string (may be implemented later for printing out trees)        
        return ''

    def print_expanded_tree(self):
        # recursively prints an expanded tree.  Uses indentation for structure
        def recurse(node,spacing):  
            if node.val is not None:          
                print(spacing+node.name+'='+str(node.val))
            else:
                print(spacing+node.name)
            if node.nextL is not None:
                recurse(node.nextL,spacing+'  ')
            if node.nextR is not None:
                recurse(node.nextR,spacing+'  ')
        recurse(self.head,'')

    def fix_constants(self,const_range):
        # fits all constants (besides x) in tree randomly with values between constant_range[0] and constant_range[1]
        node = self.head
        def recurse(node):
            if node.eq_class == 'const' and node.name != 'x':
                node.val = const_range[0]+(const_range[1]-const_range[0])*random()
            if node.nextL is not None:
                recurse(node.nextL)
            if node.nextR is not None:
                recurse(node.nextR)
        recurse(node)

    def evaluate(self,x):
        # get a y value from the equation given an x value, recursively evaluates each node up from the leaves
        node = self.head
        def get_value(node):
            if node.eq_class == 'const':
                if node.name == 'x':
                    return x
                else:
                    return node.val
            if node.eq_class == 'fn':
                return node.anon_fn(get_value(node.nextR))
            if node.eq_class == 'op':
                return node.anon_fn(get_value(node.nextL),get_value(node.nextR))
        return get_value(node)


    def flatten(self):
        # sets the string representation of the head node (tree.string_rep is equal to the equation string now)
        # returns a list of the equation elements (including parentheses) of the tree
        def get_string(node):
            if node.eq_class == 'const':
                return node.name
            elif node.eq_class == 'fn':
                return node.name +'('+ get_string(node.nextR) +')'
            else:
                return '('+ get_string(node.nextL) +')' + node.name +'('+ get_string(node.nextR) +')'
        def get_list(node):
            if node.eq_class == 'const':
                return [node.name]
            elif node.eq_class == 'fn':
                return [node.name]+['(']+ get_list(node.nextR) +[')']
            else:
                return ['(']+ get_list(node.nextL)+[')']+[node.name]+['(']+ get_list(node.nextR) +[')']
                #return get_list(node.nextL) + [node.name] + get_list(node.nextR)

        self.string_rep = get_string(self.head)
        return get_list(self.head)



















