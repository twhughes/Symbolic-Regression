import numpy as np
import matplotlib.pylab as plt
import csv

from generate_examples import *
from equation_tree import *
from encoder import *
from NN import *
#from tensorflow_test3 import *

#from keras.optimizers import Adam
#from keras.metrics import categorical_crossentropy, mean_squared_error
#from keras.models import Sequential
#from keras.layers import Dense, Activation, TimeDistributed
#from keras.layers import LSTM

#allowed = [('sin','fn',lambda x: np.sin(x)),('x','const',lambda _:True),('+','op',lambda x,y: x+y),('cos','fn',lambda x: np.cos(x)),('c','const',lambda _:True)]
allowed = [('sin','fn',lambda x: np.sin(x)),
           ('x','const',lambda _:True),
           ('+','op',lambda x,y: x+y),
           ('cos','fn',lambda x: np.cos(x))]  


# parameters for fitting to the datapoints
const_range = [-5,5]
x_range = [-5,5]
N_points = 50
N_epochs = 100
tree_depth = 5
learning_rate = 1
layer_sizes = [1,10,10,1]
activations = ['tanh','tanh','sigmoid']
N_training = 200
#perc_val = 0.3
#N_training_val = int(N_training*perc_val)

input_features, input_vectors, input_trees, losses, y_list = generate_training_examples(N_training,
                                    allowed,tree_depth=tree_depth,const_range=const_range,x_range=x_range,N_points=N_points,
                                    layer_sizes=layer_sizes,activations=activations,
                                    N_epochs=N_epochs,learning_rate=learning_rate,verbose=False)


#f = open('data/input_y_values.txt', 'w')
g = open('data/desired_equation_components.txt', 'w')
#enc = open('data/encoded_states.txt', 'w')
for i in range(N_training):
 #   y_points = y_list[i]
 #   f.write(','.join(y_points)+'\n')  # python will convert \n to os.linesep
    eq_tree = input_trees[i]
    eq_string = eq_tree.flatten()
    g.write(','.join(eq_string)+'\n')  # python will convert \n to os.linesep
#    phi = input_features[i]
#    enc.write(','.join(phi)+'\n')
#f.close()
g.close()

np.savetxt("data/input_y_values.txt", np.array(y_list), delimiter=",")
np.savetxt("data/encoded_states.txt", np.array(input_features), delimiter=",")




"""
train(input_features, input_vectors, input_trees, allowed)

verbose = False
input_features, input_vectors, input_trees, losses = generate_training_examples(N_training,
                                    allowed,tree_depth=tree_depth,const_range=const_range,x_range=x_range,N_points=N_points,
                                    layer_sizes=layer_sizes,activations=activations,
                                    N_epochs=N_epochs,learning_rate=learning_rate,verbose=False)
input_features_val, input_vectors_val, input_trees_val, losses_val = generate_training_examples(N_training_val,
                                    allowed,tree_depth=tree_depth,const_range=const_range,x_range=x_range,N_points=N_points,
                                    layer_sizes=layer_sizes,activations=activations,
                                    N_epochs=N_epochs,learning_rate=learning_rate,verbose=False)
if verbose:
    displ_index = 0
    print("String Representation  :   %s" % input_trees[displ_index].string_rep)
    print("Tree Representation    :     ")
    input_trees[displ_index].print_expanded_tree()
    print("One Hot Input Vectors  :   \n %s" % input_vectors[displ_index])
    print("Encoded Feature Vector :   \n %s" % input_features[displ_index])

inputs,outputs = process_IO_for_keras(input_features,input_vectors,N_training,allowed)
#inputs_val,outputs_val = process_IO_for_keras(input_features_val,input_vectors_val,N_training_val,allowed)

print(inputs[0,:,:])
print(outputs[0,:,:])
_,L_sequence,L_features = inputs.shape

model = Sequential([
    LSTM(L_features,batch_input_shape=(N_training,L_sequence,L_features),return_sequences=True,stateful=True),
    TimeDistributed(Dense(L_features)),
    Activation('softmax')
])

#model = Sequential([
#    LSTM(L_features,batch_input_shape=(1,L_sequence,L_features),return_sequences=True,stateful=True)
#    #TimeDistributed(Dense(L_)
#])
model.summary()
print "Inputs: {}".format(model.input_shape)
print "Outputs: {}".format(model.output_shape)
print "Actual input: {}".format(inputs.shape)
print "Actual output: {}".format(outputs.shape)
model.compile(Adam(lr=0.001),loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(np.array(inputs),np.array(outputs),batch_size=N_training,epochs=400,shuffle=True,verbose=1)

fig, ax1 = plt.subplots()
ax1.plot(history.history['acc'], 'b-')
ax1.set_xlabel('epoch num.')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('accuracy', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(history.history['loss'],'r-')
ax2.set_ylabel('loss', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.show()
input_features2, input_vectors2, input_trees2, losses2 = generate_training_examples(N_training,
                                    allowed,tree_depth=tree_depth,const_range=const_range,x_range=x_range,N_points=N_points,
                                    layer_sizes=layer_sizes,activations=activations,
                                    N_epochs=N_epochs,learning_rate=learning_rate,verbose=False)
inputs2,outputs2 = process_IO_for_keras(input_features2,input_vectors2,N_training,allowed)
print(inputs2.shape)
print(outputs2.shape)

predictions = model.predict(inputs2, batch_size=N_training, verbose=1)
predictions_binary = np.zeros((predictions.shape))
for i in range(predictions.shape[0]):
    for j in range(predictions.shape[1]):
        max_element = predictions[i,j,:].argmax()
        predictions_binary[i,j,max_element] = 1

print(input_trees2[0].string_rep)
print(predictions_binary[0,:,:])
print(outputs2[0,:,:])

"""
