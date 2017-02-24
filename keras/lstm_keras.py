import numpy as np
import nltk
import sys

import utils
from callbacks import EncDecCallback
from keras.callbacks import *
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.layers import Input, Dense
from keras.layers.core import RepeatVector
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop


class LSTMEncDec:
    def __init__(self, word_vec, word_to_index, index_to_word, weight_file=None, enc_layer_output=(32,),
                 dec_layer_output=(32,), learning_rate=0.001, sequence_len=2000, loss='mean_squared_error'):
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.sequence_len = sequence_len
        self.enc_layer_output = enc_layer_output
        self.dec_layer_output = dec_layer_output
        self.encoder = Sequential()
        self.decoder = Sequential()

        # Embedding layer should be initialized with a word-vector array and not be trained as the output relies on the same array
        self.embed = Embedding(input_dim=np.size(word_vec, 0), output_dim=np.size(word_vec, 1),
                               weights=[word_vec], trainable=False, mask_zero=True, name='Embed')

        input_layer, output_layer = self.config_processing(word_vec)

        self.model = Model(input=input_layer, output=output_layer)
        if weight_file is not None:
            self.model.load_weights(weight_file)
        self.model.compile(optimizer=RMSprop(learning_rate), loss=loss)
        # self.model.build(sequence_len)

    def config_processing(self, word_vec):
        # Configure input layer
        input_layer = Input(shape=(self.sequence_len,), name='Input')

        # Configure encoder network with the given output sizes.
        # Embedding for encoder only since decoder receives the question vector.
        self.encoder.add(self.embed)
        for el in self.enc_layer_output[:-1]:
            self.encoder.add(LSTM(el, return_sequences=True, consume_less='mem'))
        self.encoder.add(LSTM(self.enc_layer_output[-1]))  # Final LSTM layer only outputs the last vector
        self.encoder.add(RepeatVector(self.sequence_len))  # Repeat the final vector for answer input
        # Encoder outputs the question vector as a tensor with with each time-step output being the final question vector
        question_vec = self.encoder(input_layer)

        # Configure decoder network with the given output sizes.
        # Layer connecting to encoder output
        self.decoder.add(
            LSTM(self.dec_layer_output[0], input_shape=(self.sequence_len, self.enc_layer_output[-1]),
                 name='ConnectorLSTM', return_sequences=True, consume_less='mem'))
        for dl in self.dec_layer_output[1:]:
            self.decoder.add(LSTM(dl, return_sequences=True, consume_less='mem'))
        # Final layer outputting a sequence of word vectors
        self.decoder.add(LSTM(np.size(word_vec, 1), return_sequences=True, consume_less='mem'))
        output_layer = self.decoder(question_vec)
        return input_layer, output_layer

    def train(self, Xtrain, ytrain, nb_epoch, batch_size=10, queries=None):
        callback = EncDecCallback(self, queries)
        self.model.fit(Xtrain, ytrain, nb_epoch=nb_epoch, batch_size=batch_size, callbacks=[callback], verbose=1)

    def train_generator(self, Xtrain, ytrain, nb_epoch, batch_size=10, queries=None):
        callback = EncDecCallback(self, queries, True)
        total_len = np.size(ytrain, 0)
        self.model.fit_generator(utils.generate_vector_batch(Xtrain, ytrain, self.embed.get_weights()[0], total_len, batch_size),
                                 samples_per_epoch=total_len, nb_epoch=nb_epoch, callbacks=[callback],
                                 verbose=1, max_q_size=1, nb_worker=1)

    def generate_response(self, query):
        tokens = nltk.word_tokenize(query)
        indexes = [self.word_to_index[w] if w in self.word_to_index
                   else self.word_to_index[utils.UNKNOWN_TOKEN] for w in tokens]
        indexes.extend([0] * (self.sequence_len - len(indexes)))
        indexes = np.asarray(indexes, dtype=np.int32).reshape((1, self.sequence_len))
        output = self.model.predict(indexes, batch_size=1, verbose=0)
        vectors = self.embed.get_weights()[0][1:-1]
        response = []
        for word_vec in output[0]:
            word = self.index_to_word[utils.nearest_vector_index(vectors, word_vec) + 1]
            if word == utils.SENTENCE_END_TOKEN:
                break
            response.append(word)

        return ' '.join(response)


class LSTMEncDec2(LSTMEncDec):
    def __init__(self, word_vec, word_to_index, index_to_word, weight_file=None, enc_layer_output=(32,),
                 dec_layer_output=(32,), learning_rate=0.001, sequence_len=2000, loss='categorical_crossentropy'):
        super().__init__(word_vec, word_to_index, index_to_word, weight_file, enc_layer_output,
                         dec_layer_output, learning_rate, sequence_len, loss)

    def config_processing(self, word_vec):
        # Configure input layer
        input_layer = Input(shape=(self.sequence_len,), name='Input')

        # Configure encoder network with the given output sizes.
        # Embedding for encoder only since decoder receives the question vector.
        self.encoder.add(self.embed)
        for el in self.enc_layer_output[:-1]:
            self.encoder.add(LSTM(el, return_sequences=True, consume_less='mem'))
        self.encoder.add(LSTM(self.enc_layer_output[-1]))  # Final LSTM layer only outputs the last vector
        self.encoder.add(RepeatVector(self.sequence_len))  # Repeat the final vector for answer input
        # Encoder outputs the question vector as a tensor with with each time-step output being the final question vector
        question_vec = self.encoder(input_layer)

        # Configure decoder network with the given output sizes.
        # Layer connecting to encoder output
        self.decoder.add(LSTM(self.dec_layer_output[0], input_shape=(self.sequence_len, self.enc_layer_output[-1]),
                              name='ConnectorLSTM', return_sequences=True, consume_less='mem'))
        for dl in self.dec_layer_output[1:]:
            self.decoder.add(LSTM(dl, return_sequences=True, consume_less='mem'))
        # Final layer outputting a word distribution
        self.decoder.add(Dense(len(self.index_to_word), activation='softmax'))
        output_layer = self.decoder(question_vec)
        return input_layer, output_layer

    def train(self, Xtrain, ytrain, nb_epoch, batch_size=10, queries=None):
        callback = EncDecCallback(self, queries, True)
        nb_class = len(self.index_to_word)
        total_len = np.size(ytrain, 0)
        self.model.fit_generator(utils.generate_batch(Xtrain, ytrain, nb_class, total_len, batch_size), samples_per_epoch=total_len,
                                 nb_epoch=nb_epoch, callbacks=[callback], verbose=1, max_q_size=1, nb_worker=1)

    def generate_response(self, query):
        tokens = nltk.word_tokenize(query)
        indexes = [self.word_to_index[w] if w in self.word_to_index
                   else self.word_to_index[utils.UNKNOWN_TOKEN] for w in tokens]
        indexes.extend([0] * (self.sequence_len - len(indexes)))
        indexes = np.asarray(indexes, dtype=np.int32).reshape((1, self.sequence_len))
        output = self.model.predict(indexes, batch_size=1, verbose=0)
        response = []
        for word_vec in output[0]:
            word = self.index_to_word[np.argmax(word_vec[1:-1], axis=0)]
            if word == utils.SENTENCE_END_TOKEN:
                break
            response.append(word)

        return ' '.join(response)