import numpy as np
import tensorflow as tf
import sys

# set up training data
feature_vector_arr = [[1,0],
                      [0,1]]

equation_strings_arr = [['sin','(','x',')','<eoe>'],
                        ['x','+','c','<eoe>']]     # correct equation labels
#feature_vector_arr = [[1,0]]             # single input to LSTM
#equation_strings_arr = [['sin','(','x','+','c',')']]     # correct equation labels

N_feature = len(feature_vector_arr[0])
eq_dict = {'(':0,')':1,'x':2,'c':3,'sin':4,'+':5,'cos':6,'<eoe>':7}  # id the equation components
reverse_dict = {a:b for b,a in eq_dict.iteritems()}

N_vocab = len(eq_dict)
N_train = len(equation_strings_arr)
N_steps = max([len(e) for e in equation_strings_arr])
LSTM_size = 10

# turn the equation into a one-hot representation
def get_one_hot(eq_string):
    one_hot_list = []
    for i in range(N_steps):
        one_hot = np.zeros((N_vocab,1))
        if len(eq_string) > i:
            s = eq_string[i]
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
    out, _ = tf.contrib.rnn.static_rnn(lstm_cell,[feature], dtype=tf.float32)
    # apply first connected layer to output
    out = tf.reshape(out,[LSTM_size,-1])
    out = tf.add(tf.matmul(Wo,out),bo)
    # apply softmax and get max entry
#    out = tf.sigmoid(out)    
    out = tf.nn.softmax(out,dim=0)
    predict1 = tf.argmax(out)
    out_list = [out]
    for i in range(N_steps-1):
        in_state = tf.add(tf.matmul(Wi,out),bi)
        in_state = tf.reshape(in_state,[1,LSTM_size])
        out, state = tf.contrib.rnn.static_rnn(lstm_cell,[in_state], dtype=tf.float32)
        # apply first connected layer to output
        out = tf.reshape(out,[LSTM_size,-1])
        out = tf.add(tf.matmul(Wo,out),bo)
        # apply softmax and get max entry
 #       out = tf.sigmoid(out)
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
        if eq_el == '<eoe>':
            return equation
    return equation

loss = tf.constant(0.0)
#for i in range(N_train):    
#    out_list = predict(feature, lstm_cell)
#    true_out = np.array(get_one_hot(equation_strings_arr[i]))
#    loss = loss + tf.reduce_sum(tf.abs(tf.subtract(out_list,true_out)))

out_list = tf.reshape(predict(feature, lstm_cell),[1,N_steps,N_vocab])
loss = loss + tf.reduce_sum(tf.square(tf.abs(tf.subtract(out_list,target))))

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)
N_epoch = 1000

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
        losses.append(epoch_loss)
        sys.stdout.write("\repoch %s of %s.  loss: %s" % (i,N_epoch,epoch_loss))
        sys.stdout.flush()

    print("\n")

    def test_prediction(index):
        p = sess.run(out_list,feed_dict={feature:features[index]})
        eq_pred = one_hot_to_eq_str(p)
        print("supplied feature vector for : %s" % (''.join(equation_strings_arr[index])))
        print("predicted equation of       : %s" % (eq_pred))

    test_prediction(0)
    test_prediction(1)

    p = sess.run(out_list,feed_dict={feature:np.array([[1,1]])})
    eq_pred = one_hot_to_eq_str(p)
    #print("supplied feature vector for : %s" % (''.join(equation_strings_arr[index])))
    print("predicted equation of       : %s" % (eq_pred))  
      
    #p = sess.run(out_list,feed_dict={feature:features[1]})   
    #print(p) 
    #print(one_hot_to_eq_str(p))
    #print(reverse_dict)
    #print(p[0])
    #print(np.argmax(p[0]))
    #sess.run(loss,feed_dict={feature:feature_array,target:eq_one_hot})


#print(features)
#print(eq_one_hot)
