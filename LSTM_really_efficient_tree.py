import numpy as np
import tensorflow as tf
import sys
from data_loader import load_data
import pickle 
from equation_tree import *

#========Loading data from file========

fname_phi = './data/encoded_states.txt'
fname_eq = './data/desired_equation_components.txt'
fname_trees = './data/equation_trees.p'
fname_allowed = './data/allowed.p'

feature_vector_full, equation_strings_full, one_hot_list, eq_dict, reverse_dict = load_data(fname_phi,fname_eq)
equation_trees_full = pickle.load(open( fname_trees, "rb" ))
allowed = pickle.load(open( fname_allowed, "rb" ))

#========Separating training and testing data========
train_ratio = 0.7
N_total = len(feature_vector_full)
feature_vector_train = feature_vector_full[:int(N_total*train_ratio)]
feature_vector_test = feature_vector_full[int(N_total*train_ratio):N_total]
equation_strings_train = equation_strings_full[:int(N_total*train_ratio)]
equation_strings_test = equation_strings_full[int(N_total*train_ratio):N_total]
equation_trees_train = equation_trees_full[:int(N_total*train_ratio)]
equation_trees_test = equation_trees_full[int(N_total*train_ratio):N_total]

#========Get rid of parentheses========

class_dict = {name : eq_class for (name,eq_class,_) in allowed if name != ')' and name != '('}

del eq_dict[')']
del eq_dict['(']
def rename_eq_dict(eq_dict):
    index = 0
    eq_dict_new = {}
    for e in eq_dict.keys():
        eq_dict_new[e] = index
        index += 1
    return eq_dict_new
eq_dict = rename_eq_dict(eq_dict)

reverse_dict = {index : name for name,index in eq_dict.iteritems()}
class_dict['<eoe>'] = 'const'   #HACK

#========Compute and print important information for the next steps========

def compute_max_depth(trees):
    def recurse(node,depth):
        if node == None:
            return depth
        else:
            return max([recurse(node.nextL,depth+1),recurse(node.nextR,depth+1)])
    max_depth = 0
    for t in trees:
        depth = recurse(t.head,0)
        if depth > max_depth:
            max_depth = depth
    return max_depth

depth_buffer = 0
max_depth = compute_max_depth(equation_trees_full) + depth_buffer
num_elements = sum([2**i for i in range(max_depth)])
N_feature = len(feature_vector_full[0])
N_vocab = len(eq_dict)
N_train = len(equation_strings_train)
N_test = len(equation_strings_test)
N_steps = num_elements
LSTM_size = 40

print('working on %s total examples' % N_total)
print('    number of training examples : %s' % N_train)
print('    number of test examples     : %s' % N_test)
print('    considering a max depth of  : %s' % (max_depth + depth_buffer))
print('    number of equation elements : %s' % N_vocab)
print('    maximum # equation elements : %s' % N_steps)
print('    length of feature vector    : %s' % N_feature)
print('    size of LSTM states         : %s' % LSTM_size)

#========Get one-hot representations and prep data for training========

def eq_tree_to_one_hot(eq_tree, depth):
    one_hot_list = []
    def recurse(node,depth):
        if depth > 0:
            one_hot = np.zeros((N_vocab,1))
            if node is None:
                node = Node()
                one_hot[eq_dict['<eoe>'],0] = 1
                node.name = '<eoe>'
            else:
                s = node.name
                one_hot[eq_dict[s],0] = 1                
            one_hot_list.append(one_hot)
            node.one_hot = one_hot
            if node is None:
                recurse(None,depth-1)
                recurse(None,depth-1)                
            elif node.eq_class == 'op':
                recurse(node.nextL,depth-1)
                recurse(node.nextR,depth-1)
            elif node.eq_class == 'fn':
                recurse(node.nextR,depth-1)
                recurse(None,depth-1)
            else:
                recurse(None,depth-1)
                recurse(None,depth-1)  
    recurse(eq_tree.head, depth)
    return one_hot_list

features_train = [np.reshape(np.array(f),(1,N_feature)) for f in feature_vector_train]
features_test = [np.reshape(np.array(f),(1,N_feature)) for f in feature_vector_test]
features_full = [np.reshape(np.array(f),(1,N_feature)) for f in feature_vector_full]
true_one_hot_train = [np.reshape(eq_tree_to_one_hot(eq_tree,max_depth),(1,-1,N_vocab)) for eq_tree in equation_trees_train]
true_one_hot_test  = [np.reshape(eq_tree_to_one_hot(eq_tree,max_depth),(1,-1,N_vocab)) for eq_tree in equation_trees_test]
true_one_hot_full  = [np.reshape(eq_tree_to_one_hot(eq_tree,max_depth),(1,-1,N_vocab)) for eq_tree in equation_trees_full]

#========Define placeholders and variables for tensorflow graph========

# input to the first LSTM cell (the feature vector)
feature = tf.placeholder(tf.float32,[1,N_feature])
# target out values from each LSTM cell
target = tf.placeholder(tf.float32,[1,N_steps,N_vocab])

# weights and biases from LSTM to softmax
Wo = tf.Variable(tf.random_normal([N_vocab,LSTM_size]))
bo = tf.Variable(tf.zeros([N_vocab,1]))
# from softmax to next LSTM input
Wo_R = tf.Variable(tf.random_normal([N_vocab,N_vocab]))
bo_R = tf.Variable(tf.zeros([N_vocab,1]))
Wo_L = tf.Variable(tf.random_normal([N_vocab,N_vocab]))
bo_L = tf.Variable(tf.zeros([N_vocab,1]))
# weights and biases from input to LSTM
Wi = tf.Variable(tf.random_normal([LSTM_size,N_vocab]))
bi = tf.Variable(tf.zeros([LSTM_size,1]))
# vector input weights and biases from feature vector to input
Wf = tf.Variable(tf.random_normal([N_feature,N_vocab]))
bf = tf.Variable(tf.zeros([1,N_vocab]))
# define the basic lstm cell
lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_size)

#========Function to create the computational graph and compute loss========

def predict_train(input_to_LSTM, out_list, init_state, depth):
    # reformat inputs to LSTM
    input_to_LSTM = tf.add(tf.matmul(Wi,input_to_LSTM),bi)
    input_to_LSTM = tf.reshape(input_to_LSTM,[1,1,LSTM_size])
    # get outputs and states
    out, state = tf.nn.dynamic_rnn(lstm_cell,input_to_LSTM,initial_state=init_state,dtype=tf.float32)
    # get encoded outputs
    out = tf.reshape(out,[LSTM_size,-1])
    out = tf.add(tf.matmul(Wo,out),bo)
    out = tf.nn.softmax(out,dim=0) 
    out_list.append(out)
    if depth > 0:
        # create new inputs to right and left LSTMS
        out_R = tf.add(tf.matmul(Wo_R,out),bo_R)
        out_R = tf.nn.softmax(out_R,dim=0)        
        out_L = tf.add(tf.matmul(Wo_L,out),bo_L) 
        out_L = tf.nn.softmax(out_L,dim=0)
        predict_train(out_L, out_list, state, depth-1)
        predict_train(out_R, out_list, state, depth-1)

#========Function to predict an equation tree given a feature vector========
def predict_test(input_to_LSTM, init_state, node, pred_out, depth):
    # reformat inputs to LSTM
    input_to_LSTM = tf.add(tf.matmul(Wi,input_to_LSTM),bi)
    input_to_LSTM = tf.reshape(input_to_LSTM,[1,1,LSTM_size])
    # get outputs and states
    out, state = tf.nn.dynamic_rnn(lstm_cell,input_to_LSTM,initial_state=init_state,dtype=tf.float32)
    # get encoded outputs
    out = tf.reshape(out,[LSTM_size,-1])
    out = tf.add(tf.matmul(Wo,out),bo)
    out = tf.nn.softmax(out,dim=0) 
    node.one_hot = out
    pred_out.append(out)
    if depth > 0:
        # create new inputs to right and left LSTMS
        out_R = tf.add(tf.matmul(Wo_R,out),bo_R)
        out_R = tf.nn.softmax(out_R,dim=0)        
        out_L = tf.add(tf.matmul(Wo_L,out),bo_L) 
        out_L = tf.nn.softmax(out_L,dim=0)        
        node.nextL = Node()
        node.nextR = Node()
        predict_test(out_L, state, node.nextL, pred_out, depth-1)
        predict_test(out_R, state, node.nextR, pred_out, depth-1)

#========Define loss and optimizer========

print("building computational graph...")
loss = tf.constant(0.0)
input_to_LSTM = tf.reshape(tf.add(tf.matmul(feature,Wf),bf),[N_vocab,1])
predict_tree = EquationTree()
out_list = []
predict_train(input_to_LSTM, out_list, None, max_depth-1) 
out_list = tf.reshape(out_list,[1,-1,N_vocab])  
loss = tf.reduce_sum(tf.square(tf.abs(out_list-target)))

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)
N_epoch = 400

with tf.Session() as sess:
    print("")
    print("initializing tensorflow variables...")
    sess.run(tf.global_variables_initializer())
    print("")

    losses = []
    for i in range(N_epoch):
        epoch_loss = 0.0
        for m in range(N_train):
            _, loss_calc = sess.run([optimizer, loss], feed_dict={
                                                    feature:features_train[m],
                                                    target:true_one_hot_train[m]})          
        epoch_loss += loss_calc
        if i == 0:
            print("first epoch_loss = %s" % epoch_loss)
        losses.append(epoch_loss)
        sys.stdout.write("\repoch %s of %s.  loss: %s" % (i,N_epoch,epoch_loss))
        sys.stdout.flush()

    print("\n")
    print("predicting trees with max depth of %s..."%(max_depth-1))
    pred_list = []
    matches_train = 0
    matches_test = 0

    input_to_LSTM = tf.reshape(tf.add(tf.matmul(feature,Wf),bf),[N_vocab,1])
    eq_tree_predict = EquationTree()

    pred_out = []
    predict_test(input_to_LSTM, None, eq_tree_predict.head, pred_out, max_depth)

    for m in range(N_total):

        print("working on %sth test example..."%(m))

        predicted_one_hots = sess.run(pred_out,feed_dict={feature:features_full[m]})

        def load_one_hots_into_tree(node, one_hots, index_list, depth):
            index = index_list[0]
            one_hot = one_hots[index]
            node.one_hot = one_hot
            if depth > 0:
                node.nextL = Node()
                node.nextR = Node()                
                load_one_hots_into_tree(node.nextL, one_hots, [index_list[0]+1], depth-1)
                load_one_hots_into_tree(node.nextR, one_hots, [index_list[0]+1], depth-1)

        pred_tree = EquationTree()
        load_one_hots_into_tree(pred_tree.head,predicted_one_hots,[0],max_depth-1)

        def print_one_hot_tree(node):
            if node is not None:
                print(node.one_hot)
                print_one_hot_tree(node.nextR)
                print_one_hot_tree(node.nextL)
        #uncomment for debugging
        #print_one_hot_tree(pred_tree.head)
        
        def recurse_and_prune(node, depth):
            one_hot = node.one_hot
            predict = np.argmax(one_hot)
            index = int(predict)
            symbol = reverse_dict[index]
            node.name = symbol
            eq_class = class_dict[symbol]
            node.eq_class = eq_class
            if depth > 0:
                if eq_class == 'op':
                    recurse_and_prune(node.nextL, depth-1)
                    recurse_and_prune(node.nextR, depth-1)
                elif eq_class == 'fn':
                    recurse_and_prune(node.nextR, depth-1)
                    node.nextL = None
                else:
                    node.nextR = None
                    node.nextL = None

        recurse_and_prune(pred_tree.head, max_depth-1)

        eq_string = pred_tree.flatten()
        eq_string = ''.join(eq_string)

        original_equation = ''.join(equation_strings_full[m][:-1])
        print("  original equation  : %s"%(original_equation))
        print("  predicted equation : %s"%(eq_string))
        if eq_string == original_equation and m < N_train:
            matches_train += 1
        if eq_string == original_equation and m >= N_train:
            matches_test += 1            

    print("done.")
    print("Matched train = %s/%s equations for an accuracy of %s percent" % (matches_train,N_train,int(float(matches_train)/N_train*1000)/10))
    print("Matched test = %s/%s equations for an accuracy of %s percent" % (matches_test,N_test,int(float(matches_test)/N_test*1000)/10))
    print("Matched total = %s/%s equations for an accuracy of %s percent" % ((matches_train+matches_test),N_total,int(float(matches_train+matches_test)/N_total*1000)/10))

    # save the model?


