import numpy as np
import tensorflow as tf
import sys
from data_loader import load_data


fname_phi = './data/encoded_states.txt'
fname_eq = './data/desired_equation_components.txt'

feature_vector_arr, equation_strings_arr, one_hot_list, eq_dict, reverse_dict = load_data(fname_phi,fname_eq)

N_feature = len(feature_vector_arr[0])

N_vocab = len(eq_dict)
N_train = len(equation_strings_arr)
N_steps = max([len(e) for e in equation_strings_arr])
LSTM_size = 40

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


    feature = tf.add(tf.matmul(feature,Wf),bf)
    out, state = tf.contrib.rnn.static_rnn(lstm_cell,[feature], dtype=tf.float32)

    out = tf.reshape(out,[LSTM_size,-1])
    out = tf.add(tf.matmul(Wo,out),bo)

    out = tf.nn.softmax(out,dim=0)
    out_list = [out]
    print eq_dict
    #==============
    #implemented by Siddharth for semantic dictionary
    #news = tf.InteractiveSession()
    #predict = tf.argmax(out)
    #predict1 = predict.eval()
    #eq_el = reverse_dict[predict1]
    #news.close()
    #outPrev = out
    #==============

    for i in range(N_steps-1):

        input_element = tf.add(tf.matmul(Wi,out),bi)
        input_element = tf.reshape(input_element,[1,LSTM_size])
        out, state = tf.contrib.rnn.static_rnn(lstm_cell,[input_element], initial_state=state, dtype=tf.float32)

        out = tf.reshape(out,[LSTM_size,-1])
        out = tf.add(tf.matmul(Wo,out),bo)
        out = tf.nn.softmax(out,dim=0)
        #==============
        #implemented by Siddharth for semantic dictionary
        #out = check_dict(out,outPrev)
        #news = tf.InteractiveSession()
        #predict = tf.argmax(out)
        #predict1 = predict.eval()
        #eq_el = reverse_dict[predict1]
        #news.close()
        #==============

        out_list.append(out)
    return out_list

def check_dictOld(currentList,prevList):
    prevGuess = tf.argmax(prevList)
    print reverse_dict
    outTensor = [0.0 for j in range(N_vocab)]

    if prevGuess == '<eoe>':
        outTensor[eq_dict['<eoe>']] = 1

    elif prevGuess == ')':
        outTensor[eq_dict[')']] = 1
        outTensor[eq_dict['+']] = 1
        outTensor[eq_dict['*']] = 1
        outTensor[eq_dict['<eoe>']] = 1

    elif prevGuess in ['sin','cos','log','tanh']:
        outTensor[eq_dict['(']] = 1

    elif prevGuess in ['(','*','+']:
        outTensor = [1.0 for j in range(N_vocab)]
        outTensor[eq_dict[')']] = 0
        outTensor[eq_dict['+']] = 0
        outTensor[eq_dict['*']] = 0

    elif prevGuess in ['x','c']:
        outTensor[eq_dict[')']] = 1
        outTensor[eq_dict['+']] = 1
        outTensor[eq_dict['*']] = 1

    outTensor = tf.convert_to_tensor(outTensor)
    outTensor = tf.reshape(outTensor,[N_vocab,1])
    print outTensor
    print currentList
    return currentList
    #return tf.multiply(outTensor,currentList)

def check_dict(currentList,prevList):

    prevGuess = tf.argmax(prevList)
    outTensor = [0.0 for j in range(N_vocab)]

    if prevGuess == 3:
        print 'Hello'
        outTensor[eq_dict['<eoe>']] = 1

    elif prevGuess == 6:
        print 'Hello'
        outTensor[eq_dict[')']] = 1
        outTensor[eq_dict['+']] = 1
        outTensor[eq_dict['*']] = 1
        outTensor[eq_dict['<eoe>']] = 1

    elif prevGuess in [4,7,8]:
        print 'Hello'
        outTensor[eq_dict['(']] = 1

    elif prevGuess in [5,2,1]:
        print 'Hello'
        outTensor = [1.0 for j in range(N_vocab)]
        outTensor[eq_dict[')']] = 0
        outTensor[eq_dict['+']] = 0
        outTensor[eq_dict['*']] = 0

    elif prevGuess in [0,9]:
        print 'Hello'
        outTensor[eq_dict[')']] = 1
        outTensor[eq_dict['+']] = 1
        outTensor[eq_dict['*']] = 1

    outTensor = tf.convert_to_tensor(outTensor)
    outTensor = tf.reshape(outTensor,[N_vocab,1])
    print outTensor
    print currentList
    return currentList
    #return tf.multiply(outTensor,currentList)
def one_hot_to_eq_str(one_hot_list):
    one_hot_list = one_hot_list[0]  # need to get 0th element since only one training example in practice
    N = len(one_hot_list)
    equation = ''
    eq_el = ''
    for i in range(N):
        one_hot_allowed = one_hot_list[i]
        #one_hot_allowed = check_dict(one_hot_list[i],eq_el)
        #Implement Gibbs sampling:
        #cdf = np.cumsum(one_hot_allowed)
        prediction = np.argmax(one_hot_allowed)

        eq_el = reverse_dict[prediction]
        equation += eq_el
    return equation

alpha = 0.5*0
beta = 0.5*0
gamma = 0.6*0
loss = tf.constant(0.0)
out_list = predict(feature, lstm_cell)
out_list_tensor = tf.reshape(out_list,[1,N_steps,N_vocab])
#======Mismatch loss======
loss = loss + tf.reduce_sum(tf.square(tf.abs(tf.subtract(out_list_tensor,target))))
#======Heuristic losses======
A = [0 for j in range(N_vocab)]
A[eq_dict['+']] = 1
A[eq_dict['*']] = 1
A[eq_dict['<eoe>']] = 1
A[eq_dict[')']] = 1
A[eq_dict['(']] = 1
A = tf.constant([A],dtype=tf.float32)
pred = tf.reshape(out_list[0],[N_vocab,1])
loss += alpha*tf.reduce_sum(tf.matmul(A,pred))

count = 0
X = [0 for j in range(N_vocab)]
Y = [0 for j in range(N_vocab)]
X[eq_dict['(']]=1
Y[eq_dict[')']]=1
X = tf.constant([X],dtype=tf.float32)
Y = tf.constant([Y],dtype=tf.float32)

for j in range(N_steps):
    pred = tf.reshape(out_list[j],[N_vocab,1])
    count += tf.reduce_sum(tf.matmul(X,pred))
    count -= tf.reduce_sum(tf.matmul(Y,pred))
loss += beta*np.abs(count)

C = [[1 for j in range(N_vocab)] for i in range(N_vocab)]
#'''
okList = {'<eoe>':['<eoe>'],'cos':['('],'sin':['('],'tanh':['('],\
'+':['x','sin','cos','tanh'],'*':['x','sin','cos','tanh'],\
'(':['x','cos','sin','tanh'],')':['+','*',')'],'x':['+','*',')']}
#'''
'''
okList = {'<eoe>':['<eoe>'],'cos':['('],'sin':['('],'tanh':['('],\
'+':['x','sin','cos','tanh'],'(':['x','cos','sin','tanh'],\
')':['+',')'],'x':['+',')']}
'''
'''
okList = {'<eoe>':['<eoe>'],'sin':['('],\
'(':['x','sin'],')':[')'],'x':[')']}
'''
for sym in okList:
    index = eq_dict[sym]
    for ok in okList[sym]:
        allowed = eq_dict[ok]
        C[index][allowed] = 0

C = tf.constant(C,dtype=tf.float32)

for j in range(1,N_steps):
    prevPred = tf.reshape(out_list[j-1],[N_vocab,1])
    currentPred = tf.reshape(out_list[j],[N_vocab,1])
    loss += gamma*tf.reduce_sum(tf.matmul(tf.transpose(prevPred),tf.matmul(C,currentPred)))
#=============================


optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)
N_epoch = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    losses = []
    for i in range(N_epoch):
        epoch_loss = 0.0
        for m in range(N_train):
            _, loss_calc, out_list_calc = sess.run([optimizer, loss, out_list_tensor], \
                                                            feed_dict={ feature:features[m],
                                                                         target:eq_one_hot[m]})
            epoch_loss += loss_calc
        losses.append(epoch_loss)
        sys.stdout.write("\repoch %s of %s.  loss: %s" % (i,N_epoch,epoch_loss))
        sys.stdout.flush()

    print("\n")

    def test_prediction(index):
        p = sess.run(out_list_tensor,feed_dict={feature:features[index]})
        eq_pred = one_hot_to_eq_str(p)
        suppliedString = ''.join(equation_strings_arr[index]).replace('<eoe>','')
        predictedString = eq_pred.replace('<eoe>','')
        print("supplied feature vector for : %s" % (suppliedString))
        print("predicted equation of       : %s" % (predictedString))

        if predictedString==suppliedString:
            return 1
        else:
            return 0

    correctPreds = 0

    for j in range(len(features)):
        correctPreds += test_prediction(j)

    print 'Number of correct predictions: %d out of %d' %(correctPreds, N_train)
