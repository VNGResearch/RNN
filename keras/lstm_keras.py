from operator import mul
from functools import reduce

import numpy as np
import nltk
import keras.backend as K
import utils

from callbacks import EncDecCallback
from keras.callbacks import *
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.layers import Input, Dense, TimeDistributed
from keras.layers.core import RepeatVector
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, RMSprop, Adadelta


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
        self.compile(learning_rate, loss)

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
        self.decoder.add(TimeDistributed(Dense(np.size(word_vec, 1), activation='linear')))
        # self.decoder.add(LSTM(np.size(word_vec, 1), return_sequences=True, consume_less='mem'))
        output_layer = self.decoder(question_vec)
        return input_layer, output_layer

    def compile(self, learning_rate, loss):
        self.model.compile(optimizer=Adadelta(), loss=loss, metrics=['mean_absolute_error'], sample_weight_mode='temporal')

    def train(self, Xtrain, ytrain, nb_epoch, output_mask, batch_size=10, queries=None):
        callback = EncDecCallback(self, queries)
        self.model.fit(Xtrain, ytrain, nb_epoch=nb_epoch, batch_size=batch_size, callbacks=[callback], verbose=1)

    def train_generator(self, Xtrain, ytrain, nb_epoch, Xval=None, yval=None, batch_size=10, queries=None):
        callback = EncDecCallback(self, queries)
        total_len = np.size(ytrain, 0)
        if Xval is None or yval is None:
            self.model.fit_generator(utils.generate_vector_batch(Xtrain, ytrain, self.embed.get_weights()[0], total_len, batch_size),
                                     samples_per_epoch=total_len, nb_epoch=nb_epoch, callbacks=[callback],
                                     verbose=1, max_q_size=1, nb_worker=1)
        else:
            self.model.fit_generator(utils.generate_vector_batch(Xtrain, ytrain, self.embed.get_weights()[0], total_len, batch_size),
                                     samples_per_epoch=total_len, nb_epoch=nb_epoch, callbacks=[callback], nb_val_samples=100,
                                     verbose=1, max_q_size=1, nb_worker=1,
                                     validation_data=utils.generate_vector_batch(Xval, yval, self.embed.get_weights()[0], 100, 1))

    def generate_response(self, query):
        tokens = nltk.word_tokenize(query.lower())[:self.sequence_len]
        indices = [self.word_to_index[w] if w in self.word_to_index
                   else self.word_to_index[utils.UNKNOWN_TOKEN] for w in tokens]
        indices.extend([0] * (self.sequence_len - len(indices)))
        indices = np.asarray(indices, dtype=np.int32).reshape((1, self.sequence_len))
        output = self.model.predict(indices, batch_size=1, verbose=0)
        vectors = self.embed.get_weights()[0]
        response = []
        for word_vec in output[0]:
            word = self.index_to_word[utils.nearest_vector_index(vectors, word_vec)]
            if word == utils.MASK_TOKEN:
                continue
            elif word == utils.SENTENCE_END_TOKEN:
                break
            response.append(word)

        return ' '.join(response)


class LSTMEncDec2(LSTMEncDec):
    def __init__(self, word_vec, word_to_index, index_to_word, output_mask=None, weight_file=None, enc_layer_output=(32,),
                 dec_layer_output=(32,), learning_rate=0.001, sequence_len=2000, loss='categorical_crossentropy'):
        self.mask = output_mask
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
        self.decoder.add(TimeDistributed(Dense(len(self.index_to_word), activation='softmax')))
        output_layer = self.decoder(question_vec)
        return input_layer, output_layer

    def compile(self, learning_rate, loss):
        self.model.compile(optimizer=RMSprop(learning_rate), loss=loss, metrics=[self.categorical_accuracy], sample_weight_mode='temporal')

    def train(self, Xtrain, ytrain, nb_epoch, Xval=None, yval=None, batch_size=10, queries=None):
        callback = EncDecCallback(self, queries, True)
        nb_class = len(self.index_to_word)
        total_len = np.size(ytrain, 0)
        if Xval is None or yval is None:
            self.model.fit_generator(utils.generate_batch(Xtrain, ytrain, self.mask, nb_class, total_len, batch_size), samples_per_epoch=total_len,
                                     nb_epoch=nb_epoch, callbacks=[callback], verbose=1, max_q_size=1, nb_worker=1)
        else:
            self.model.fit_generator(utils.generate_batch(Xtrain, ytrain, self.mask, nb_class, total_len, batch_size),
                                     samples_per_epoch=total_len, nb_epoch=nb_epoch, callbacks=[callback], verbose=1,
                                     validation_data=utils.generate_batch(Xval, yval, self.mask, nb_class, 100, 1),
                                     max_q_size=1, nb_worker=1, nb_val_samples=100)

    def generate_response(self, query):
        tokens = nltk.word_tokenize(query.lower())[:self.sequence_len]
        indices = [self.word_to_index[w] if w in self.word_to_index
                   else self.word_to_index[utils.UNKNOWN_TOKEN] for w in tokens]
        indices.extend([0] * (self.sequence_len - len(indices)))
        indices = np.asarray(indices, dtype=np.int32).reshape((1, self.sequence_len))
        output = self.model.predict(indices)
        out_idx = np.argmax(output, axis=2)
        response = []
        # noinspection PyTypeChecker
        for idx in out_idx[0]:
            word = self.index_to_word[idx]
            if word == utils.MASK_TOKEN:
                continue
            elif word == utils.SENTENCE_END_TOKEN:
                break
            response.append(word)
        return ' '.join(response)

    def categorical_accuracy(self, y_true, y_pred):
        if self.mask is not None:
            eval_shape = (reduce(mul, y_true.shape[:-1]), y_true.shape[-1])
            y_true_ = K.reshape(y_true, eval_shape)
            y_pred_ = K.reshape(y_pred, eval_shape)
            msk = K.variable(self.mask, dtype=np.float32, name='mask')
            flat_mask = K.flatten(msk)
            comped = K.equal(K.argmax(y_true_, axis=-1),
                             K.argmax(y_pred_, axis=-1))
            # not sure how to do this in tensor flow
            good_entries = flat_mask.nonzero()[0]
            return K.mean(K.gather(comped, good_entries))

        else:
            return K.mean(K.equal(K.argmax(y_true, axis=-1),
                                  K.argmax(y_pred, axis=-1)))
