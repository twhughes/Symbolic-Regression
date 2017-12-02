import numpy as np
import tensorflow as tf
import sys

# set up training data
feature_vector_arr = [[1,0],
                      [0,1]]             # single input to LSTM
equation_strings_arr = [['sin','(','x','+','c',')'],
                        ['cos','(','x','+','c',')']]     # correct equation labels
#feature_vector = [1,0]             # single input to LSTM
#equation_strings = ['sin','(','x','+','c',')']     # correct equation labels

N_steps = max([len(equation_strings) for equation_strings in equation_strings_arr])
N_feature = len(feature_vector_arr[0])
eq_dict = {'(':0,')':1,'x':2,'c':3,'sin':4,'+':5,'cos':6}  # id the equation components
N_vocab = len(eq_dict)
N_train = len(equation_strings_arr)
# turn the equation into a one-hot representation
def get_one_hot(eq_string):
    one_hot_list = []
    for s in eq_string:
        one_hot = np.zeros((N_vocab,1))
        one_hot[eq_dict[s],0] = 1
        one_hot_list.append(one_hot)
    return one_hot_list
# turn the equation into a one-hot representation and reshape for TF
one_hot_list = []
phi_list = []
for i in range(N_train):
    one_hot_list.append(np.array(get_one_hot(equation_strings_arr[i])))
    phi_list.append(np.reshape(np.array(feature_vector_arr[i],dtype=float),(1,N_feature)))
phi_list = np.reshape(np.array(phi_list),(N_train,N_feature))
one_hot_list = np.reshape(np.array(one_hot_list),(N_train,N_steps,N_vocab))

# input to the first LSTM cell (the feature vector)
feature = tf.placeholder(tf.float32,[None,N_feature])
# target out values from each LSTM cell
target = tf.placeholder(tf.float32,[None,N_steps,N_vocab])
print(feature)
print(target)
# output weights and biases (to softmax)
Wo = tf.Variable(tf.random_normal([N_vocab,N_feature]))
bo = tf.Variable(tf.zeros([N_vocab,1]))
# output weights and biases (to softmax)
Wi = tf.Variable(tf.random_normal([N_feature,N_vocab]))
bi = tf.Variable(tf.zeros([N_feature,1]))

# define the basic lstm cell
lstm_cell = tf.contrib.rnn.BasicLSTMCell(N_train, N_feature)

def predict(features, lstm_cell):
    # first output from feeding the feature vector
    #initial_state = tf.zeros([1, lstm_cell.state_size])
    #initial_input = tf.zeros([1, lstm_cell.state_size])
    out, _ = tf.contrib.rnn.static_rnn(lstm_cell,[features], dtype=tf.float32)
    # apply first connected layer to output
    out = tf.reshape(out,[N_feature,-1])
    out = tf.add(tf.matmul(Wo,out),bo)
    # apply softmax and get max entry
    out = tf.nn.softmax(out,dim=0)
    predict1 = tf.argmax(out)
    out_list = [out]
    for i in range(N_steps-1):
        in_state = tf.add(tf.matmul(Wi,out),bi)
        in_state = tf.reshape(in_state,[-1,N_feature])
        out, state = tf.contrib.rnn.static_rnn(lstm_cell,[in_state], dtype=tf.float32)
        # apply first connected layer to output
        out = tf.reshape(out,[N_feature,-1])
        out = tf.add(tf.matmul(Wo,out),bo)
        # apply softmax and get max entry
        out = tf.nn.softmax(out,dim=0)
        predict = tf.argmax(out)
        out_list.append(out)
    return out_list


# note: there is some strange error about variable scope.
# I need to set lstm_cell.reuse = None, run and then set = True to get it to work.
# Has to do with re-running the script in a notebook but I haven't figured out how to fix yet.

loss = tf.constant(0.0)
target_list = tf.split(target[i],num_or_size_splits=N_steps)
out_list = predict(feature, lstm_cell)
for i in range(N_steps):
    loss = loss + tf.reduce_sum(tf.abs(tf.subtract(out_list,target_list)))

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)

N_epoch = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    losses = []
    for i in range(N_epoch):
        _, loss_calc, out_list_calc = sess.run([optimizer, loss, out_list], \
                                                        feed_dict={ feature:phi_list,target:one_hot_list})
        losses.append(loss_calc)
        sys.stdout.write("\rloss: %s" % loss_calc)
        sys.stdout.flush()
    #sess.run(out_list,feed_dict={feature:phi_list})
    #sess.run(loss,feed_dict={feature:feature_array,target:eq_one_hot})

print(out_list_calc)
print(eq_one_hot)
