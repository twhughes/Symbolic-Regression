import numpy as np
from generate_examples import *
from equation_tree import *
from encoder import *
import matplotlib.pylab as plt
from RNN import *
from NN import *
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM

#allowed = [('sin','fn',lambda x: np.sin(x)),('x','const',lambda _:True),('+','op',lambda x,y: x+y),('cos','fn',lambda x: np.cos(x)),('c','const',lambda _:True)]
allowed = [('sin','fn',lambda x: np.sin(x)),
           ('x','const',lambda _:True),
           ('+','op',lambda x,y: x+y),
           ('cos','fn',lambda x: np.cos(x))]           

const_range = [-5,5]
x_range = [-5,5]
N_points = 50
N_epochs = 100
tree_depth = 6
learning_rate = 1
layer_sizes = [1,10,10,1]
activations = ['tanh','tanh','sigmoid']
N_training = 200
verbose = False

input_features, input_vectors, input_trees, losses = generate_training_examples(N_training,
                                    allowed,tree_depth=tree_depth,const_range=const_range,x_range=x_range,N_points=N_points,
                                    layer_sizes=layer_sizes,activations=activations,
                                    N_epochs=N_epochs,learning_rate=learning_rate,verbose=False)

# NOTE:
# At this point, things are ready for the RNN.
# input_features is a list of numpy arrays (one per training example) containing the feature vectors phi from the NN fit
# input_vectors is a list of numpy arrays (one per training example) containing the desired outputs from the LSTM
# need to train a model that can take input_features -> input_vectors
#

if verbose:
    displ_index = 0
    print("String Representation  :   %s" % input_trees[displ_index].string_rep)
    print("Tree Representation    :     ")
    input_trees[displ_index].print_expanded_tree()
    print("One Hot Input Vectors  :   \n %s" % input_vectors[displ_index])
    print("Encoded Feature Vector :   \n %s" % input_features[displ_index])


inputs,outputs = process_IO_for_keras(input_features,input_vectors,N_training,allowed)

_,L_sequence,L_features = inputs.shape

model = Sequential([
    LSTM(L_features,batch_input_shape=(N_training,L_sequence,L_features),return_sequences=True,stateful=True),
])
model.summary()
print "Inputs: {}".format(model.input_shape)
print "Outputs: {}".format(model.output_shape)
print "Actual input: {}".format(inputs.shape)
print "Actual output: {}".format(outputs.shape)
model.compile(Adam(lr=0.0001),loss='mean_squared_error', metrics=['accuracy'])
model.fit(np.array(inputs),np.array(outputs),batch_size=N_training,epochs=10000,shuffle=True,verbose=2)



