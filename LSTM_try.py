
import numpy as np
import tensorflow as tf
import csv

# load in data from file
fname_eq = './data/desired_equation_components.txt'
fname_phi = './data/encoded_states.txt'
equation_strings = []
with open(fname_eq, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\n')
    for row in reader:
        equation_strings.append(row[0].split(','))
phi_list = []
with open(fname_phi, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\n')
    for row in reader:
        str_phi = row[0].split(',')
        phi_list.append([float(s) for s in str_phi])
phi_list = np.array(phi_list)
def create_dictionaries(equation_strings):
    dictionary = {}
    reverse_dictionary = {}
    index = 0
    for equation in equation_strings:
        for element in equation:
            if element not in dictionary.keys():
                dictionary[element] = index
                reverse_dictionary[index] = element
                index += 1
    return dictionary, reverse_dictionary

def eq_strings_to_one_hot(equation_strings, dictionary):
    one_hot_list = []
    for eq_str in equation_strings:
        M = len(eq_str)
        N = len(dictionary)
        one_hot = np.zeros((M,N))
        for eq_index, eq_element in enumerate(eq_str):
            one_hot_index = dictionary[eq_element]
            one_hot[eq_index,one_hot_index] = 1
        one_hot_list.append(one_hot)
    return one_hot_list

dictionary, reverse_dictionary = create_dictionaries(equation_strings)
one_hot_list = eq_strings_to_one_hot(equation_strings, dictionary)
one_hot_list = np.array(one_hot_list)

num_classes = len(dictionary)
num_training = len(equation_strings)
num_features = len(phi_list[0])
max_len_equation = max([len(e) for e in equation_strings])
print(dictionary)
print(one_hot_list[0])
print(equation_strings[0])

features = tf.placeholder(tf.float32,[None,num_features])
true_labels = tf.placeholder(tf.float32,[None,max_len_equation+1,num_classes])

# output weights and biases (to softmax)
Wo = tf.Variable(tf.random_normal([num_classes,num_features]))
bo = tf.Variable(tf.zeros([num_classes,1]))
# output weights and biases (to softmax)
Wi = tf.Variable(tf.random_normal([num_features,num_classes]))
bi = tf.Variable(tf.zeros([num_features,1]))

lstm_cell = tf.contrib.rnn.BasicLSTMCell([num_features])

def predict(features_in, lstm_cell):
    print(features_in)
    out, _ = tf.contrib.rnn.static_rnn(lstm_cell,[features_in], dtype=tf.float32)
    # apply first connected layer to output
    out = tf.reshape(out,[N_feature,-1])
    out = tf.add(tf.matmul(Wo,out),bo)
    # apply softmax and get max entry
    #out = tf.nn.softmax(out,dim=1)
    predict1 = tf.argmax(out)
    out_list = [out]
    for i in range(N_steps-1):
        in_state = tf.add(tf.matmul(Wi,out),bi)
        in_state = tf.reshape(in_state,[1,N_feature])
        out, state = tf.contrib.rnn.static_rnn(lstm_cell,[in_state], dtype=tf.float32)
        # apply first connected layer to output
        out = tf.reshape(out,[num_features,-1])
        out = tf.add(tf.matmul(Wo,out),bo)
        # apply softmax and get max entry
        #out = tf.nn.softmax(out,dim=0)
        predict = tf.argmax(out)
        out_list.append(out)
    return out_list

loss = tf.constant(0.0)
print(one_hot_list)
out_list = predict(features, lstm_cell)
loss = tf.reduce_sum(tf.abs(tf.subtract(out_list,one_hot_list)))

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
    sess.run(out_list,feed_dict={feature:phi_list})



















