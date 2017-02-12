import numpy as np
from utils import *
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.layers import Input
from keras.layers.core import RepeatVector
from keras.layers.embeddings import Embedding


class LSTMEncDec:
    def __init__(self, word_vec, word_to_index, index_to_word, enc_layer_output=(32,), dec_layer_output=(32,), sequence_len=2000):
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.embed = Embedding(input_dim=np.size(word_vec, 0), output_dim=np.size(word_vec, 1), weights=[word_vec])

        # Configure input layer
        input_layer = Input(shape=(sequence_len,), name='Input')

        # Configure encoder network with the given output sizes.
        self.encoder = Sequential()
        # Embedding for encoder only since decoder receives the question vector.
        self.encoder.add(self.embed)
        for el in enc_layer_output:
            self.encoder.add(LSTM(el, return_sequences=True))
        self.encoder.add(RepeatVector(sequence_len))  # Repeat the final vector for answer input
        # Encoder outputs the question vector as a tensor with with each time-step output being the final question vector
        question_vec = self.encoder(input_layer)

        # Configure decoder network with the given output sizes.
        self.decoder = Sequential()
        # Layer connecting to encoder output
        self.decoder.add(LSTM(dec_layer_output[0], input_shape=(sequence_len, enc_layer_output[-1]), name='ConnectorLSTM', return_sequences=True))
        for dl in dec_layer_output[1:]:
            self.decoder.add(LSTM(dl, return_sequences=True))
        # Final layer outputting a sequence of word vectors
        self.decoder.add(LSTM(np.size(word_vec, 1), return_sequences=True))
        output_layer = self.decoder(question_vec)

        self.model = Model(input=input_layer, output=output_layer)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.model.build(sequence_len)
