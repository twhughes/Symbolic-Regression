import numpy as np
import tensorflow as tf
import sys
from data_loader import load_data
import pickle 
from equation_tree import *

fname_phi = './data/encoded_states.txt'
fname_eq = './data/desired_equation_components.txt'
fname_trees = './data/equation_trees.p'
fname_allowed = './data/allowed.p'

feature_vector_arr, equation_strings_arr, one_hot_list, eq_dict, reverse_dict = load_data(fname_phi,fname_eq)
equation_trees_arr = pickle.load(open( fname_trees, "rb" ))
allowed = pickle.load(open( fname_allowed, "rb" ))

class_dict = {name : eq_class for (name,eq_class,_) in allowed }

N_feature = len(feature_vector_arr[0])
N_vocab = len(eq_dict)
N_train = len(equation_strings_arr)
N_steps = max([len(e) for e in equation_strings_arr])
LSTM_size = 24

print('working on %s examples' % N_train)
print('    number of equation elements : %s' % N_vocab)
print('    maximum equation length     : %s' % N_steps)
print('    length of feature vector    : %s' % N_feature)
print('    size of LSTM states         : %s' % LSTM_size)

# turn the equation into a one-hot representation
def eq_string_one_hot(eq_string):
    one_hot_list = []
    for i in range(N_steps):
        one_hot = np.zeros((N_vocab,1))
        if i < len(eq_string):
            s = eq_string[i]
            one_hot[eq_dict[s],0] = 1
        else:
            s = '<eoe>'
            one_hot[eq_dict[s],0] = 1            
        one_hot_list.append(one_hot)
    return one_hot_list

def one_hot_to_eq_str(one_hot_list):
    one_hot_list = one_hot_list[0]  # need to get 0th element since only one training example in practice
    N = len(one_hot_list)
    equation = ''
    for i in range(N):
        prediction = np.argmax(one_hot_list[i])
        eq_el = reverse_dict[prediction]
        equation += eq_el
    return equation

def eq_tree_to_one_hot(eq_tree):
    one_hot_list = []
    def recurse(node):
        one_hot = np.zeros((N_vocab,1))
        s = node.name
        one_hot[eq_dict[s],0] = 1
        one_hot_list.append(one_hot)
        if node.eq_class == 'op':
            recurse(node.nextL)
            recurse(node.nextL)
        elif node.eq_class == 'fn':
            recurse(node.nextR)        
    recurse(eq_tree.head)
    L = len(one_hot_list)
    for i in range(N_steps-L):
        one_hot = np.zeros((N_vocab,1))
        one_hot[eq_dict['<eoe>'],0] = 1
        one_hot_list.append(one_hot)
    return one_hot_list

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

# turn the equation into a one-hot representation and reshape for TF
features = [np.reshape(np.array(f),(1,N_feature)) for f in feature_vector_arr]
eq_one_hot = [eq_tree_to_one_hot(eq_tree) for eq_tree in equation_trees_arr]
eq_one_hot = [np.reshape(np.array(e),(1,N_steps,N_vocab)) for e in eq_one_hot]

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

def predict_train(input_to_LSTM, train_tree_node, init_state, out_list):
    # reformat inputs to LSTM
    input_to_LSTM = tf.add(tf.matmul(Wi,input_to_LSTM),bi)
    input_to_LSTM = tf.reshape(input_to_LSTM,[1,1,LSTM_size])
    # get outputs and states
    with tf.device('/cpu:0'):
        out, state = tf.nn.dynamic_rnn(lstm_cell,input_to_LSTM,initial_state=init_state,dtype=tf.float32)
    # get encoded outputs
    out = tf.reshape(out,[LSTM_size,-1])
    out = tf.add(tf.matmul(Wo,out),bo)
    out = tf.nn.softmax(out,dim=0) 

    out_list.append(out)
    # create new inputs to right and left LSTMS
    out_R = tf.add(tf.matmul(Wo_R,out),bo_R)
    out_R = tf.nn.softmax(out_R,dim=0)        
    out_L = tf.add(tf.matmul(Wo_L,out),bo_L) 
    out_L = tf.nn.softmax(out_L,dim=0)

    if train_tree_node.eq_class == 'op':
        predict_train(out_L, train_tree_node.nextL, state, out_list)
        predict_train(out_R, train_tree_node.nextR, state, out_list)
    elif train_tree_node.eq_class == 'fn':
        predict_train(out_R, train_tree_node.nextR, state, out_list)

def predict_test(input_to_LSTM, init_state, node, depth):
    # reformat inputs to LSTM
    input_to_LSTM = tf.add(tf.matmul(Wi,input_to_LSTM),bi)
    input_to_LSTM = tf.reshape(input_to_LSTM,[1,1,LSTM_size])
    # get outputs and states
    with tf.device('/cpu:0'):
        out, state = tf.nn.dynamic_rnn(lstm_cell,input_to_LSTM,initial_state=init_state,dtype=tf.float32)
    # get encoded outputs
    out = tf.reshape(out,[LSTM_size,-1])
    out = tf.add(tf.matmul(Wo,out),bo)
    out = tf.nn.softmax(out,dim=0) 

    node.val = out
    # create new inputs to right and left LSTMS
    out_R = tf.add(tf.matmul(Wo_R,out),bo_R)
    out_R = tf.nn.softmax(out_R,dim=0)        
    out_L = tf.add(tf.matmul(Wo_L,out),bo_L) 
    out_L = tf.nn.softmax(out_L,dim=0)

    if depth > 0:
        node.nextL = Node()
        node.nextR = Node()        
        predict_test(out_L, state, node.nextL, depth-1)
        predict_test(out_R, state, node.nextR, depth-1)


def loss_fn(m):
    loss = tf.constant(0.0)
    input_to_LSTM = tf.reshape(tf.add(tf.matmul(feature,Wf),bf),[N_vocab,1])
    train_tree_node = equation_trees_arr[m]
    out_list = []
    predict_train(input_to_LSTM,train_tree_node.head,None,out_list)
    # pad with <eoe> for remaining
    for _ in range(N_steps-len(out_list)):
        one_hot = np.zeros((N_vocab,1))
        one_hot[eq_dict['<eoe>'],0] = 1
        out_list.append(one_hot)
    new_out_list = tf.reshape(out_list,[1,N_steps,N_vocab])
    #loss = loss + tf.reduce_sum(tf.abs(tf.subtract(new_out_list,target)))
    loss = loss + tf.reduce_sum(tf.square(tf.abs(tf.subtract(new_out_list,target))))    
    return loss

optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
def optimizer_fn(m):
    return optimizer.minimize(loss_fn(m))

loss_fns = []
optimizer_fns = []
print("")
for i in range(N_train):
    sys.stdout.write("\rsetting up tree for training example %s of %s..." % (i,N_train))
    sys.stdout.flush()
    loss_fns.append(loss_fn(i))
    optimizer_fns.append(optimizer_fn(i))
print("")

N_epoch = 300
depth_buffer = 2

with tf.Session() as sess:
    print("")
    print("initializing tensorflow variables...")
    sess.run(tf.global_variables_initializer())
    print("")

    losses = []
    # TO DO: precompute LSTM trees separately (reuse parameters)
    #        then train each epoch without needing to reconstruct tree for each training example each epoch
    for i in range(N_epoch):
        epoch_loss = 0.0
        for m in range(N_train):
            loss_m = loss_fns[m]
            opt_m = optimizer_fns[m]
            _, loss_calc = sess.run([opt_m, loss_m], feed_dict={
                                                    feature:features[m],
                                                    target:eq_one_hot[m]})          
            epoch_loss += loss_calc
        if i == 0:
            print("first epoch_loss = %s" % epoch_loss)
        losses.append(epoch_loss)
        sys.stdout.write("\repoch %s of %s.  loss: %s" % (i,N_epoch,epoch_loss))
        sys.stdout.flush()

    print("\n")

    max_depth = compute_max_depth(equation_trees_arr) + depth_buffer
    print("predicting trees with max depth of %s..."%(max_depth-1))
    pred_list = []
    matches = 0
    for m in range(N_train):
        print("working on %sth test example..."%(m))

        input_to_LSTM = tf.reshape(tf.add(tf.matmul(feature,Wf),bf),[N_vocab,1])
        eq_tree_predict = EquationTree()
        predict_test(input_to_LSTM, None, eq_tree_predict.head, max_depth)

        def recurse_and_prune(node):
            one_hot = node.val
            predict = tf.argmax(one_hot)
            index = sess.run(predict, feed_dict={feature:features[m]})
            index = int(index)
            symbol = reverse_dict[index]
            node.name = symbol
            eq_class = class_dict[symbol]
            node.eq_class = eq_class

            if node.nextL is not None and node.nextR is not None:
                if eq_class == 'op':
                    recurse_and_prune(node.nextL)
                    recurse_and_prune(node.nextR)
                elif eq_class == 'fn':
                    recurse_and_prune(node.nextR)
                    node.nextL = None
                else:
                    node.nextR = None
                    node.nextL = None

        recurse_and_prune(eq_tree_predict.head)
        eq_string = eq_tree_predict.flatten()
        eq_string = ''.join(eq_string)
        original_equation = ''.join(equation_strings_arr[m][:-1])
        print("  original equation  : %s"%(original_equation))
        print("  predicted equation : %s"%(eq_string))
        if eq_string == original_equation:
            matches += 1
    print("done.  Matched %s/%s equations for an accuracy of %s percent" % (matches,N_train,int(float(matches)/N_train*1000)/10))

        #for e in eq_list:
        #    index = sess.run(e, feed_dict={feature:features[m]})
        #    eq_symbol_list.append(dict[eq_symbol_list])












"""
    def test_prediction(index):
        p = sess.run(out_list,feed_dict={feature:features[index]})
        eq_pred = one_hot_to_eq_str(p)
        eq_true = ''.join(equation_strings_arr[index])
        L = len(eq_true)
        print("supplied feature vector for : %s" % eq_true[:L])
        print("predicted equation of       : %s" % eq_pred[:L])
        return eq_true[:L] == eq_pred[:L]            

    matches = 0
    for i in range(N_train):
        matches += test_prediction(i)

    print('predicted correctly on %s/%s training examples:  %s percent accuracy'%(matches,N_train,int(float(matches)/N_train*1000.0)/10.0))
"""
    #p = sess.run(out_list,feed_dict={feature:features[1]})   
    #print(p) 
    #print(one_hot_to_eq_str(p))
    #print(reverse_dict)
    #print(p[0])
    #print(np.argmax(p[0]))
    #sess.run(loss,feed_dict={feature:feature_array,target:eq_one_hot})


#print(features)
#print(eq_one_hot)
