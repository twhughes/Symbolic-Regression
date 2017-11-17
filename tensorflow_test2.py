'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow..
Next word prediction after n_input words learned from text file.
A story is automatically generated if the predicted word is fed back as input.

Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
from utils import *

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

# Text file containing words for training
training_file = 'belling_the_cat.txt'

def train(input_features, input_vectors, input_trees, allowed):

    #get mapping of eq_string '+','x',etc. to index in one_hot (and vice-versa)
    dictionary, reverse_dictionary = allowed_to_dict(allowed)
    eq_vocab_size = len(dictionary)  #length of the resulting one-hot vectors

    # Parameters
    max_eq_length = 40                    # default length of input
    learning_rate = 0.001                   # optimization learning rate
    training_iters = 5000                   # training iterations?
    display_step = 1000                     # display step?

    # number of units in RNN cell
    n_hidden = len(input_features[0])    # length of feature vector

    # tf Graph input
    x = tf.placeholder("float64", [None, max_eq_length, n_hidden])
    # Y is a tensor with first dimension: # training, second dimension: one hot vector output
    y = tf.placeholder("float64", [None, max_eq_length, eq_vocab_size])

    # RNN output node weights and biases
    weights = {
    # weights 'in' multiplies one hot input and gives embedding like feature vector
        'in': tf.Variable(tf.random_normal([n_hidden, eq_vocab_size],dtype=tf.float64)),
    # weights 'out' multiplies LSTM output and gives softmax-like output prediction       
        'out': tf.Variable(tf.random_normal([eq_vocab_size,n_hidden],dtype=tf.float64))
    }
    biases = {
        'in': tf.Variable(tf.random_normal([n_hidden,1],dtype=tf.float64)),
    # see above.. but for biases               
        'out': tf.Variable(tf.random_normal([eq_vocab_size,1],dtype=tf.float64))
    }

    def RNN(x, weights, biases):
        x = tf.split(x,max_eq_length,1)
        for i in range(len(x)):
            x[i] = tf.reshape(x[i],[1,n_hidden])
        # 2-layer LSTM, each layer has n_hidden units.
        # Average Accuracy= 95.20% at 50k iter
        # rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

        # 1-layer LSTM with n_hidden units but with lower accuracy.
        # Average Accuracy= 90.60% 50k iter
        # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
        rnn_cell = rnn.BasicLSTMCell(n_hidden)
        # generate prediction
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float64)
        # return a list of softmax predictions at each cell
        return [tf.nn.softmax(tf.matmul(weights['out'],tf.transpose(outputs[i])) + biases['out']) for i in range(max_eq_length)]

    pred = RNN(x, weights, biases)
    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as session:
        session.run(init)
        step = 0
        acc_total = 0
        loss_total = 0

        writer.add_graph(session.graph)

        while step in range(training_iters):
            # get phi to send to LSTM
            feature_vector = input_features[step]
            # generate list of symbol keys of length max_eq_length-1 to send to LSTM
            key_list = input_trees[step].flatten()
            num_keys = len(key_list)
            key_list = key_list + ['' for _ in range(max_eq_length-num_keys-1)]
            print('maximum equation length of %s with original key list length of %s, new key list of   %s' % (max_eq_length, num_keys, len(key_list)))
            symbols_in_keys = np.reshape(np.array(key_list), [-1, max_eq_length-1, 1])
            # reshape to [1, n_input]
            inputs = np.zeros((eq_vocab_size, max_eq_length-1))
            for index,key in enumerate(key_list):
                inputs[dictionary[key],index]=1
            inputs = tf.split(inputs,max_eq_length-1,1)
            # get new input to LSTM with extra elements chopped off
            # NOTE: should we add a special <end of equation> element?
            # directly add feature vector to first input
            new_inputs = [tf.reshape(np.array(feature_vector),[n_hidden,-1])]
            for i,xi in enumerate(inputs):
                new_inputs.append(tf.sigmoid(tf.matmul(weights['in'],xi) + biases['in']))
            inputs = new_inputs            
            symbols_out_onehot = np.zeros((eq_vocab_size, max_eq_length))
            for index,key in enumerate(key_list):
                symbols_out_onehot[dictionary[key],index]=1
            print(inputs)
            print(tf.split(symbols_out_onehot,max_eq_length,1))
            symbols_out_onehot = tf.split(symbols_out_onehot,max_eq_length,1)
            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                    feed_dict={  x: inputs,
                                                                 y: symbols_out_onehot})
            loss_total += loss
            acc_total += acc
            if (step+1) % display_step == 0:
                print("Iter= " + str(step+1) + ", Average Loss= " + \
                      "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                      "{:.2f}%".format(100*acc_total/display_step))
                acc_total = 0
                loss_total = 0
                symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                symbols_out = training_data[offset + n_input]
                symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        print("Optimization Finished!")
        print("Elapsed time: ", elapsed(time.time() - start_time))
        print("Run on command line.")
        print("\ttensorboard --logdir=%s" % (logs_path))
        print("Point your web browser to: http://localhost:6006/")
        while True:
            prompt = "%s words: " % n_input
            sentence = input(prompt)
            sentence = sentence.strip()
            words = sentence.split(' ')
            if len(words) != n_input:
                continue
            try:
                symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
                for i in range(32):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                    onehot_pred = session.run(pred, feed_dict={x: keys})
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                    sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(onehot_pred_index)
                print(sentence)
            except:
                print("Word not in dictionary")
