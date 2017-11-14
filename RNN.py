from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras import Model
from keras.utils.vis_utils import plot_model

def build_LSTM_model(allowed, input_features, input_trees):
    # configure
    #num_encoder_tokens = 71
    num_decoder_tokens = len(allowed)
    latent_dim = len(input_features[0])
    # Define an input sequence and process it.
    #encoder_inputs = Input(shape=(None, num_encoder_tokens))
    #encoder = LSTM(latent_dim, return_state=True)
    #encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    #encoder_states = [state_h, state_c]
    # Set up the decoder, using `encoder_states` as initial state.
    feature_vec = Input(shape=(None, num_decoder_tokens))
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([feature_vec, decoder_inputs], decoder_outputs)
    # plot the model
    plot_model(model, to_file='model.png', show_shapes=True)