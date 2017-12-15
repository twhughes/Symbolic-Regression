'''
Accelerating Symbolic Regression with Deep Learning
By Tyler Hughes, Siddharth Buddhiraju and Rituraj
For CS 221, Fall 2017-2018

This file reads examples generated from generate_examples.py, and trains a
decoder(LSTM) to minimize L2 loss
on the generated equation against the correct equation.
Then, it predicts on the test dataset and returns statistics.
'''

import numpy as np
import tensorflow as tf
import sys
from data_loader import load_data

#============Read examples from file ========================
max_depth = 2

fname_phi = './data/encoded_states_d'+str(max_depth)+'.txt'
fname_eq = './data/desired_equation_components_d'+str(max_depth)+'.txt'

feature_vector_arr, equation_strings_arr, one_hot_list, eq_dict,\
 reverse_dict = load_data(fname_phi,fname_eq)

#========Separating training and testing data========
feature_vector_full = feature_vector_arr
equation_strings_full = equation_strings_arr
train_ratio = 0.95
N_total = len(feature_vector_arr)
feature_vector_test = feature_vector_arr[int(N_total*train_ratio):N_total]
equation_strings_test = equation_strings_arr[int(N_total*train_ratio):N_total]
feature_vector_train = feature_vector_arr[0:int(N_total*train_ratio)]
equation_strings_train = equation_strings_arr[0:int(N_total*train_ratio)]

#=================Setting up LSTM parameters==========
N_feature = len(feature_vector_train[0])
N_vocab = len(eq_dict)
N_train = len(equation_strings_train)
N_steps = max([len(e) for e in equation_strings_train])
LSTM_size = 20
N_epoch = 300

print('working on %s examples' % N_train)
print('    number of equation elements : %s' % N_vocab)
print('    maximum equation length     : %s' % N_steps)
print('    length of feature vector    : %s' % N_feature)
print('    size of LSTM states         : %s' % LSTM_size)

#===========Functions================================
# turn the equation into a one-hot representation
def get_one_hot(eq_string):
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

def predictTraintime(feature, target, lstm_cell):


    feature = tf.add(tf.matmul(feature,Wf),bf)
    out, state = tf.contrib.rnn.static_rnn(lstm_cell,[feature], dtype=tf.float32)

    #cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.8)  <- Tried but did not improve results
    #out, state = tf.contrib.rnn.static_rnn(cell,[feature], dtype=tf.float32)

    out = tf.reshape(out,[LSTM_size,-1])
    out = tf.add(tf.matmul(Wo,out),bo)


    out = tf.nn.softmax(out,dim=0)
    out_list = [out] #out_list shifted above softmax for cross entropy

    for i in range(N_steps-1):

        in_prev = tf.reshape(target[0,i,:],[N_vocab,1])
        input_element = tf.add(tf.matmul(Wi,in_prev),bi)
        input_element = tf.reshape(input_element,[1,LSTM_size])
        #out, state = tf.contrib.rnn.static_rnn(cell,[input_element], initial_state=state, dtype=tf.float32)
        out, state = tf.contrib.rnn.static_rnn(lstm_cell,[input_element], initial_state=state, dtype=tf.float32)

        out = tf.reshape(out,[LSTM_size,-1])
        out = tf.add(tf.matmul(Wo,out),bo)

        out = tf.nn.softmax(out,dim=0)
        out_list.append(out) #out_list shifted above softmax for cross entropy

    return out_list

def predictTesttime(feature, lstm_cell):


    feature = tf.add(tf.matmul(feature,Wf),bf)
    out, state = tf.contrib.rnn.static_rnn(lstm_cell,[feature], dtype=tf.float32)

    out = tf.reshape(out,[LSTM_size,-1])
    out = tf.add(tf.matmul(Wo,out),bo)

    out = tf.nn.softmax(out,dim=0)
    out_list = [out]

    for i in range(N_steps-1):

        input_element = tf.add(tf.matmul(Wi,out),bi)
        input_element = tf.reshape(input_element,[1,LSTM_size])
        out, state = tf.contrib.rnn.static_rnn(lstm_cell,[input_element], initial_state=state, dtype=tf.float32)

        out = tf.reshape(out,[LSTM_size,-1])
        out = tf.add(tf.matmul(Wo,out),bo)
        out = tf.nn.softmax(out,dim=0)

        out_list.append(out)

    return out_list

def one_hot_to_eq_str(one_hot_list):
    one_hot_list = one_hot_list[0]  # need to get 0th element since only one training example in practice
    N = len(one_hot_list)
    equation = ''
    eq_el = ''
    for i in range(N):
        one_hot_allowed = one_hot_list[i]
        prediction = np.argmax(one_hot_allowed)
        eq_el = reverse_dict[prediction]
        equation += eq_el

    return equation

#===========Setting up objects for LSTM==========
# turn the equation into a one-hot representation and reshape for TF
features = [np.reshape(np.array(f),(1,N_feature)) for f in feature_vector_train]
eq_one_hot = [np.reshape(np.array(get_one_hot(e)),(1,N_steps,N_vocab)) for e in equation_strings_train]

# input to the first LSTM cell (the feature vector)
feature = tf.placeholder(tf.float32,[1,N_feature])
# desired out values from each LSTM cell
target = tf.placeholder(tf.float32,[1,N_steps,N_vocab])

# output weights and biases (to softmax)
Wo = tf.Variable(tf.random_normal([N_vocab,LSTM_size]))
bo = tf.Variable(tf.zeros([N_vocab,1]))
# output weights and biases (to softmax)
Wi = tf.Variable(tf.random_normal([LSTM_size,N_vocab]))
bi = tf.Variable(tf.zeros([LSTM_size,1]))
# output weights and biases (to softmax)
Wf = tf.Variable(tf.random_normal([N_feature,LSTM_size]))
bf = tf.Variable(tf.zeros([1,LSTM_size]))

# define the basic lstm cell
lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_size)
#==================================================

loss = tf.constant(0.0)

#=========Outlists at training and testing time=====
out_list = predictTraintime(feature, target, lstm_cell)
out_list_tensor = tf.reshape(out_list,[1,N_steps,N_vocab])

out_list_run = predictTesttime(feature, lstm_cell)
out_list_run_tensor = tf.reshape(out_list_run,[1,N_steps,N_vocab])

#======L2 Loss======
loss = loss + tf.reduce_sum(tf.square(tf.subtract(out_list_tensor,target)))
#======Cross entropy with Logits======
#loss = loss + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=target,logits=\
#out_list_tensor))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# turn the equation into a one-hot representation and reshape for TF
features_train = [np.reshape(np.array(f),(1,N_feature)) for f in\
 feature_vector_train]
features_test = [np.reshape(np.array(f),(1,N_feature)) for f in\
 feature_vector_test]
eq_one_hot_train = [np.reshape(np.array(get_one_hot(e)),(1,N_steps,N_vocab))\
 for e in equation_strings_train]

with tf.Session() as sess:

    #======Training the NN+LSTM on training data======
    sess.run(tf.global_variables_initializer())
    losses = []
    for i in range(N_epoch):
        epoch_loss = 0.0
        for m in range(N_train):
            _, loss_calc, out_list_calc = sess.run([optimizer, loss, \
            out_list_tensor], feed_dict={ feature:features_train[m],\
             target:eq_one_hot_train[m]})

            epoch_loss += loss_calc

        losses.append(epoch_loss)
        sys.stdout.write("\repoch %s of %s.  loss: %s" % (i,N_epoch,epoch_loss))
        sys.stdout.flush()

    print("\n")
    #======Test error on the training (seen) data======
    def trainError(index):
        p = sess.run(out_list_run_tensor,feed_dict=\
        {feature:features_train[index]})
        eq_pred = one_hot_to_eq_str(p)
        suppliedString = ''.join(equation_strings_train[index]).replace('<eoe>','')
        predictedString = eq_pred.replace('<eoe>','')
        #print '--'
        #print("supplied feature vector for : %s" % (suppliedString))
        #print("predicted equation of       : %s" % (predictedString))

        if (suppliedString == 'x') and (predictedString=='x'):
            return (0,1)
        elif (suppliedString == 'c') and (predictedString=='c'):
            return (0,1)
        elif predictedString==suppliedString:
            return (1,0)

        return (0,0)
    #======Test error on the test (unseen) data======
    def testError(index):
        p = sess.run(out_list_run_tensor,feed_dict={feature:features_test[index]})
        eq_pred = one_hot_to_eq_str(p)
        suppliedString = ''.join(equation_strings_test[index]).replace('<eoe>','')
        predictedString = eq_pred.replace('<eoe>','')
        print '--'
        print("supplied feature vector for : %s" % (suppliedString))
        print("predicted equation of       : %s" % (predictedString))

        if (suppliedString == 'x') and (predictedString=='x'):
            return (0,1)
        elif (suppliedString == 'c') and (predictedString=='c'):
            return (0,1)
        elif predictedString==suppliedString:
            return (1,0)

        return (0,0)

    #==================Console output========================
    print 'Testing on test data:'
    correctPreds = 0
    correctPredsX = 0
    for j in range(len(features_test)):
        output = testError(j)
        correctPreds += output[0]
        correctPredsX += output[1]
    print 'Number of correct "x/c" predictions: %d' %correctPredsX
    print 'Number of correct predictions excluding "x/c": %d' %correctPreds
    print 'Total %d out of %d' %(correctPreds+correctPredsX,\
    len(feature_vector_test))

    print ("\n")
    print 'Now on original training data:'

    correctPreds = 0
    for j in range(len(features_train)):
        output = trainError(j)
        correctPreds += output[0]
        correctPredsX += output[1]
    print 'Number of correct "x/c" predictions: %d' %correctPredsX
    print 'Number of correct predictions excluding "x/c": %d' %correctPreds
    print 'Total %d out of %d' %(correctPreds+correctPredsX,len(feature_vector_train))

    new_examples = [''.join(ex).replace('<eoe','') for ex in \
    equation_strings_test if not (ex in equation_strings_train)]

    print 'New functions were: ', new_examples

    #============Optional: Save epoch loss to file===============
    g = open('data/seq_1_1500ex_L2_300ep.txt', 'w')
    for e in losses:
        g.write(str(e)+'\n')
    g.close()
