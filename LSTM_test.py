import numpy as np
import matplotlib.pylab as plt

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM

N_time_steps = 100
N_features = 1
N_examples = 100

distribution = lambda x: np.sin(x)

inputs = np.zeros((N_examples,N_time_steps,N_features))
outputs = np.zeros((N_examples,N_time_steps,N_features))
xpoints = np.linspace(0,2*np.pi,N_time_steps)
noise = 0.1


for i in range(N_examples):
    xpoints_i = xpoints
    ypoints_i = distribution(xpoints)
    for j in range(N_time_steps):
        xpoints_i[j] = xpoints[j]+2*noise*(np.random.random()-0.5) 
        ypoints_i[j] = distribution(xpoints_i[j])+2*noise*(np.random.random()-0.5)
    inputs[i,:,0] = ypoints_i
    outputs[i,:,0] = distribution(xpoints)
    plt.plot(inputs[i,:,0],outputs[i,:,0])

plt.show()
model = Sequential([
    LSTM(N_features,batch_input_shape=(N_examples,N_time_steps,N_features),return_sequences=True,stateful=True)
])

model.summary()
print "Inputs: {}".format(model.input_shape)
print "Outputs: {}".format(model.output_shape)
print "Actual input: {}".format(inputs.shape)
print "Actual output: {}".format(outputs.shape)
model.compile(Adam(lr=0.1),loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(inputs,outputs,batch_size=N_examples,epochs=100,shuffle=True,verbose=2)
