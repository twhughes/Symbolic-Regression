import numpy as np
import tensorflow as tf
import sys
from data_loader import load_data

# set up training data
#feature_vector_arr = [[1,0],
#                      [1,0],
#                      [0,1]]

#equation_strings_arr = [['sin','(','x','+','c',')','<eoe>'],
#                        ['cos','(','x',')','<eoe>'],
#                        ['tan','(','c',')','<eoe>']]
#                        ['cos','(','x',')','<eoe>','<eoe>','<eoe>']]
#feature_vector_arr = [[1,0]]             # single input to LSTM
#equation_strings_arr = [['sin','(','x','+','c',')']]     # correct equation labels

fname_phi = './data/encoded_states.txt'
fname_eq = './data/desired_equation_components.txt'

feature_vector_arr, equation_strings_arr, one_hot_list, eq_dict, reverse_dict = load_data(fname_phi,fname_eq)

N_feature = len(feature_vector_arr[0])

N_vocab = len(eq_dict)
N_train = len(equation_strings_arr)
N_steps = max([len(e) for e in equation_strings_arr])
LSTM_size = 44

print('working on %s examples' % N_train)
print('    number of equation elements : %s' % N_vocab)
print('    maximum equation length     : %s' % N_steps)
print('    length of feature vector    : %s' % N_feature)
print('    size of LSTM states         : %s' % LSTM_size)

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


# turn the equation into a one-hot representation and reshape for TF
features = [np.reshape(np.array(f),(1,N_feature)) for f in feature_vector_arr]
eq_one_hot = [np.reshape(np.array(get_one_hot(e)),(1,N_steps,N_vocab)) for e in equation_strings_arr]

# input to the first LSTM cell (the feature vector)
feature = tf.placeholder(tf.float32,[1,N_feature])
# target out values from each LSTM cell
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
def predict(feature, lstm_cell):
    #print(feature)
    # first output from feeding the feature vector
    feature = tf.add(tf.matmul(feature,Wf),bf)
    out, state = tf.contrib.rnn.static_rnn(lstm_cell,[feature], dtype=tf.float32)
    # apply first connected layer to output
    out = tf.reshape(out,[LSTM_size,-1])
    out = tf.add(tf.matmul(Wo,out),bo)
    # apply softmax and get max entry
    #out = tf.sigmoid(out)    
    out = tf.nn.softmax(out,dim=0)
    predict = tf.argmax(out)
    out_list = [out]
    for i in range(N_steps-1):

        input_element = tf.add(tf.matmul(Wi,out),bi)
        input_element = tf.reshape(input_element,[1,LSTM_size])
        out, state = tf.contrib.rnn.static_rnn(lstm_cell,[input_element], initial_state=state, dtype=tf.float32)
        # apply first connected layer to output
        out = tf.reshape(out,[LSTM_size,-1])
        out = tf.add(tf.matmul(Wo,out),bo)
        # apply softmax and get max entry
        #out = tf.sigmoid(out)
        out = tf.nn.softmax(out,dim=0)
        predict = tf.argmax(out)
        out_list.append(out)
    return out_list

def one_hot_to_eq_str(one_hot_list):
    one_hot_list = one_hot_list[0]  # need to get 0th element since only one training example in practice
    N = len(one_hot_list)
    equation = ''
    for i in range(N):
        prediction = np.argmax(one_hot_list[i])
        eq_el = reverse_dict[prediction]
        equation += eq_el
    return equation

loss = tf.constant(0.0)
out_list = tf.reshape(predict(feature, lstm_cell),[1,N_steps,N_vocab])
loss = loss + tf.reduce_sum(tf.square(tf.abs(tf.subtract(out_list,target))))

optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)
N_epoch = 2000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    losses = []
    for i in range(N_epoch):
        epoch_loss = 0.0
        for m in range(N_train):
            _, loss_calc, out_list_calc = sess.run([optimizer, loss, out_list], \
                                                            feed_dict={ feature:features[m],
                                                                         target:eq_one_hot[m]})
            epoch_loss += loss_calc
        if i == 0:
            print("first epoch_loss = %s" % epoch_loss)
        losses.append(epoch_loss)
        sys.stdout.write("\r    epoch %s of %s.  loss: %s" % (i,N_epoch,epoch_loss))
        sys.stdout.flush()

    print("\n")

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

    #p = sess.run(out_list,feed_dict={feature:features[1]})   
    #print(p) 
    #print(one_hot_to_eq_str(p))
    #print(reverse_dict)
    #print(p[0])
    #print(np.argmax(p[0]))
    #sess.run(loss,feed_dict={feature:feature_array,target:eq_one_hot})


#print(features)
#print(eq_one_hot)
