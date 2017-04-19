import numpy as np
import nltk
import keras.backend as K
import theano
import theano.tensor as T
import utils

from recurrentshop import RecurrentContainer, LSTMCell
from seq2seq import LSTMDecoderCell
from callbacks import EncDecCallback
from keras.callbacks import *
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.layers import Input, Dense, TimeDistributed
from keras.layers.core import RepeatVector
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, RMSprop, Adadelta
from theano.ifelse import ifelse
from functools import reduce


class LSTMEncDec:
    """
    Output type: 0 for word vector output/similarity inference, 1 for softmax word distribution output.
    Decoder type: 0 for non-readout LSTM decoder, 1 for recurrentshop's readout decoder.
    """
    def __init__(self, word_vec, word_to_index, index_to_word, weight_file=None, enc_layer_output=(32,),
                 dec_layer_output=(32,), learning_rate=0.001, sequence_len=2000, directory='.',
                 out_type=0, decoder_type=0):
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.sequence_len = sequence_len
        self.directory = directory
        self.enc_layer_output = enc_layer_output
        self.dec_layer_output = dec_layer_output
        self.decoder_type = decoder_type
        self.out_type = out_type
        if out_type == 0:
            loss = 'mean_squared_error'
        elif out_type == 1:
            loss = 'categorical_crossentropy'
        else:
            raise ValueError('Invalid output type %s.' % self.out_type)
        self.encoder = Sequential()
        self.decoder = Sequential()
        self.embed = None
        self.batch_size = 0

        input_layer, output_layer = self.config_model(word_vec)

        self.model = Model(input=input_layer, output=output_layer)
        if weight_file is not None:
            self.model.load_weights(weight_file)
        self.compile(learning_rate, loss)

    """
    Creates the encoder-decoder structure and returns the symbolic input and output
    """
    def config_model(self, word_vec):
        if self.out_type == 0:
            train_embed = False
        else:
            train_embed = True

        # Configure input layer
        input_layer = Input(shape=(self.sequence_len,), name='Input')

        # Embedding layer should be initialized with a word-vector array and not be trained as the output relies on the same array
        self.embed = Embedding(input_dim=np.size(word_vec, 0), output_dim=np.size(word_vec, 1),
                               weights=[word_vec], trainable=train_embed, mask_zero=True, name='Embed')

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
        if self.decoder_type == 0:
            self.decoder.add(LSTM(self.dec_layer_output[0], input_shape=(self.sequence_len, self.enc_layer_output[-1]),
                                  name='ConnectorLSTM', return_sequences=True, consume_less='mem'))
            for dl in self.dec_layer_output[1:]:
                self.decoder.add(LSTM(dl, return_sequences=True, consume_less='mem'))
        elif self.decoder_type == 1:
            # Using recurrentshop's container with readout
            container = RecurrentContainer(readout=True, return_sequences=True,
                                           output_length=self.dec_layer_output[-1])
            container.add(LSTMCell(output_dim=self.dec_layer_output[0],
                                   input_dim=self.enc_layer_output[-1]))
            for dl in self.dec_layer_output[1:]:
                container.add(LSTMCell(output_dim=dl))
            container.add(LSTMCell(output_dim=self.enc_layer_output[-1]))  # Output must be compatible with input for merging
            self.decoder.add(container)
        else:
            raise ValueError('Invalid decoder type %s.' % self.decoder_type)

        # Final layer outputting a sequence of word vectors
        if self.out_type == 0:
            self.decoder.add(TimeDistributed(Dense(np.size(word_vec, 1), activation='linear')))
        else:
            self.decoder.add(TimeDistributed(Dense(len(self.index_to_word), activation='softmax')))
        output_layer = self.decoder(question_vec)

        return input_layer, output_layer

    def compile(self, learning_rate, loss):
        if self.out_type == 0:
            metrics = ['mean_absolute_error']
        else:
            metrics = [self.categorical_acc]
        self.model.compile(optimizer=RMSprop(lr=learning_rate), loss=loss, metrics=metrics,
                           sample_weight_mode='temporal')

    """
    Uses a generator to decompress labels from integers to hot-coded vectors batch-by-batch to save memory.
    See utils.generate_batch().
    """
    def train(self, Xtrain, ytrain, nb_epoch, Xval=None, yval=None, train_mask=None, val_mask=None, batch_size=10,
              queries=None):
        self.batch_size = batch_size
        callback = EncDecCallback(self, queries, True)
        logger = CSVLogger(self.directory + '/epochs.csv')
        nb_class = len(self.index_to_word)
        total_len = np.size(ytrain, 0)

        if self.out_type == 0:
            generator = utils.generate_vector_batch
        else:
            generator = utils.generate_batch

        if Xval is None or yval is None:
            self.model.fit_generator(generator(Xtrain, ytrain, self.embed.get_weights()[0], train_mask, nb_class, total_len, batch_size),
                                     samples_per_epoch=total_len,nb_worker=1,
                                     nb_epoch=nb_epoch, callbacks=[callback, logger], verbose=1, max_q_size=1)
        else:
            self.model.fit_generator(generator(Xtrain, ytrain, self.embed.get_weights()[0], train_mask, nb_class, total_len, batch_size),
                                     samples_per_epoch=total_len, nb_epoch=nb_epoch, callbacks=[callback, logger],
                                     verbose=1, max_q_size=1, nb_worker=1, nb_val_samples=Xval.shape[0],
                                     validation_data=generator(Xval, yval, self.embed.get_weights()[0], val_mask, nb_class, Xval.shape[0], batch_size))

    """
    Pre-processes a raw query string and return a response string 
    """
    def generate_response(self, query):
        tokens = nltk.word_tokenize(query.lower())[:self.sequence_len]
        indices = [self.word_to_index[w] if w in self.word_to_index
                   else self.word_to_index[utils.UNKNOWN_TOKEN] for w in tokens]
        indices.extend([0] * (self.sequence_len - len(indices)))
        indices = np.asarray(indices, dtype=np.int32).reshape((1, self.sequence_len))
        output = self.model.predict(indices, batch_size=1, verbose=0)
        vectors = self.embed.get_weights()[0]
        response = []

        if self.out_type == 0:
            for word_vec in output[0]:
                word = self.index_to_word[utils.nearest_vector_index(vectors, word_vec)]
                if word == utils.MASK_TOKEN:
                    continue
                elif word == utils.SENTENCE_END_TOKEN:
                    break
                response.append(word)
        else:
            out_idx = np.argmax(output, axis=2)
            # noinspection PyTypeChecker
            for idx in out_idx[0]:
                word = self.index_to_word[idx]
                if word == utils.MASK_TOKEN:
                    continue
                elif word == utils.SENTENCE_END_TOKEN:
                    response.append(word)
                    break
                response.append(word)

        return ' '.join(response)

    """
    Generates a list of top candidates for each word position given a raw string query.
    Only applies for softmax model.
    """
    def generate_candidates(self, query, top=3):
        tokens = nltk.word_tokenize(query.lower())[:self.sequence_len]
        indices = [self.word_to_index[w] if w in self.word_to_index
                   else self.word_to_index[utils.UNKNOWN_TOKEN] for w in tokens]
        indices.extend([0] * (self.sequence_len - len(indices)))
        indices = np.asarray(indices, dtype=np.int32).reshape((1, self.sequence_len))
        output = self.model.predict(indices, batch_size=1, verbose=0)
        vectors = self.embed.get_weights()[0]
        response, candidates = [], []

        if self.out_type == 0:
            for word_vec in output[0]:
                word = self.index_to_word[utils.nearest_vector_index(vectors, word_vec)]
                if word == utils.MASK_TOKEN:
                    continue
                elif word == utils.SENTENCE_END_TOKEN:
                    break
                response.append(word)
        else:
            out_idx = utils.k_largest_idx(output, top)
            # noinspection PyTypeChecker
            for ca in out_idx[0]:
                word = self.index_to_word[ca[0]]
                if word == utils.MASK_TOKEN:
                    continue
                elif word == utils.SENTENCE_END_TOKEN:
                    response.append(word)
                    break
                response.append(word)
                candidates.append([self.index_to_word[c] for c in ca])

        return ' '.join(response), candidates

    """
    Custom ad-hoc categorical accuracy for masked output (requires Theano backend).
    Dynamically detects masked characters and ignores them in calculation.
    """
    def categorical_acc(self, y_true, y_pred):
        p = self.word_to_index[utils.MASK_TOKEN]
        token = np.zeros((len(self.index_to_word)), dtype=np.float32)
        token[p] = 1
        t = K.variable(token)

        # Configure masking function
        def iterate(a):
            d2, u2 = theano.scan(fn=mask, sequences=a)
            return d2

        def mask(w):
            return ifelse(T.eq(w, t).all(), T.zeros(1), T.ones(1))

        mask, u1 = theano.scan(fn=iterate, sequences=y_true)

        eval_shape = (reduce(T.mul, y_true.shape[:-1]), y_true.shape[-1])
        y_true_ = K.reshape(y_true, eval_shape)
        y_pred_ = K.reshape(y_pred, eval_shape)
        flat_mask = K.flatten(mask)
        comped = K.equal(K.argmax(y_true_, axis=-1),
                         K.argmax(y_pred_, axis=-1))
        # not sure how to do this in tensor flow
        good_entries = flat_mask.nonzero()[0]
        return K.mean(K.gather(comped, good_entries))

    def log(self, string='', out=True):
        f = open(self.directory + '/log.txt', mode='at')

        if out:
            print(string)
        print(string, file=f)
        f.close()

    def perplexity(self, y_true, y_pred):
        dist = K.sum(y_true * y_pred, axis=2)
        length = K.variable(self.sequence_len)
        p = K.pow(K.variable(2), K.variable(0) - K.sum(T.log2(dist), axis=1)/length)
        return K.mean(p)

    def bleu_score(self, y_true, y_pred):
        pass

