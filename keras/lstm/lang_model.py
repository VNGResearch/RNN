import pickle
import sys

import logging
from keras.callbacks import *
from keras.layers import Dense, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop

import lstm.tokens as tokens
from lstm.callbacks import LangModelCallback
from utils import commons


class LSTMLangModel:
    def __init__(self, word_vec, word_to_index, index_to_word, weight_file=None,
                 learning_rate=0.001, sequence_len=2000, directory='./models/LM_debug',
                 dropout=0.0, outputs=(32,)):
        self.word_vec = word_vec
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.sequence_len = sequence_len
        self.directory = directory
        self.outputs = outputs
        self.dropout = dropout

        self.model = Sequential()
        self.embed = Embedding(input_dim=np.size(word_vec, 0), output_dim=np.size(word_vec, 1),
                               weights=[word_vec], trainable=True, mask_zero=True, name='Embed',
                               input_shape=(self.sequence_len,))
        self.model.add(self.embed)
        for lo in outputs:
            self.model.add(LSTM(lo, implementation=1, return_sequences=True, dropout=self.dropout))
        self.model.add(TimeDistributed(Dense(len(self.index_to_word), activation='softmax')))

        if weight_file is not None:
            self.model.load_weights(weight_file)

        self.model.compile(RMSprop(lr=learning_rate), 'categorical_crossentropy',
                           sample_weight_mode='temporal', metrics=[])

    def train(self, Xtrain, ytrain, nb_epoch, Xval=None, yval=None, train_mask=None, val_mask=None,
              batch_size=10):
        """
            Uses a generator to decompress labels from integers to hot-coded vectors batch-by-batch to save memory.
            See utils.commons.generate_batch().
        """
        callback = LangModelCallback(self)
        logger = CSVLogger(self.directory + '/epochs.csv')
        nb_class = len(self.index_to_word)
        total_len = np.size(ytrain, 0)

        generator = commons.generate_batch

        if Xval is None or yval is None:
            self.model.fit_generator(
                generator(Xtrain, ytrain, self.embed.get_weights()[0], train_mask, nb_class, total_len, batch_size),
                steps_per_epoch=total_len / batch_size, nb_worker=1,
                epochs=nb_epoch, callbacks=[callback, logger], verbose=1, max_q_size=1)
        else:
            self.model.fit_generator(
                generator(Xtrain, ytrain, self.embed.get_weights()[0], train_mask, nb_class, total_len, batch_size),
                steps_per_epoch=total_len / batch_size, epochs=nb_epoch, callbacks=[callback, logger],
                verbose=1, max_q_size=1, workers=1, validation_steps=Xval.shape[0] / batch_size,
                validation_data=generator(Xval, yval, self.embed.get_weights()[0], val_mask, nb_class, Xval.shape[0],
                                          batch_size))

    def predict(self, query_tokens):
        out_pos = len(query_tokens) - 1
        indices = [self.word_to_index[w] if w in self.word_to_index
                   else self.word_to_index[tokens.UNKNOWN_TOKEN] for w in query_tokens]
        indices.extend([0] * (self.sequence_len - len(indices)))
        indices = np.asarray(indices, dtype=np.int32).reshape((1, self.sequence_len))

        output = self.model.predict(indices, batch_size=1, verbose=0)
        return output[0][out_pos]

    def log(self, string='', out=True):
        f = open(self.directory + '/log.txt', mode='at')

        if out:
            print(string)
        print(string, file=f)
        f.close()

    def save(self):
        f1 = self.directory + '/weights.hdf5'
        f2 = self.directory + '/config.pkl'
        f3 = self.directory + '/dictionary.npz'

        self.model.save_weights(f1)
        config = {
            'seq_len': self.sequence_len,
            'word_vec_dim': np.shape(self.word_vec),
            'outputs': self.outputs,
            'dropout': self.dropout
        }
        pickle.dump(config, open(f2, 'wb'), pickle.HIGHEST_PROTOCOL)
        np.savez(f3, wit=self.word_to_index, itw=self.index_to_word,
                 wv=self.word_vec)
        logging.info('Saved model to %s' % self.directory)

    @staticmethod
    def load(directory):
        f1 = directory + '/weights.hdf5'
        f2 = directory + '/config.pkl'
        f3 = directory + '/dictionary.npz'

        logging.info('Loading model from %s...' % directory)
        try:
            config = pickle.load(open(f2, 'rb'))

            npz_file = np.load(f3)
            word_to_index, index_to_word, word_vec = npz_file["wit"].reshape(1)[0], npz_file["itw"], npz_file[
                "wv"].reshape(config['word_vec_dim'])

            logging.info('Done.')
            return LSTMLangModel(word_vec, word_to_index, index_to_word, weight_file=f1,
                                 sequence_len=config.get('seq_len', 2000), directory=directory,
                                 outputs=config.get('outputs', (32,)), dropout=config.get('dropout', 0.0))
        except FileNotFoundError:
            print('One or more model files cannot be found. Terminating...')
            sys.exit()
