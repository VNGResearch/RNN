import numpy as np
import datetime
import pickle
from lstm_keras import *


SENTENCE_END_TOKEN = '$SENTENCE_END'
UNKNOWN_TOKEN = '$UNKNOWN'
MASK_TOKEN = '<MASK>'
DIRECTORY = './models/LSTM_%s' % datetime.date.today().isoformat()


def log(string=''):
    f = open(DIRECTORY + '/.log', mode='at')
    print(string)
    print(string, file=f)
    f.close()


def nearest_vector(array, value):
    return array[nearest_vector_index(array, value)]


def nearest_vector_index(array, value):
    return np.sum(np.abs(array-value), axis=1).argmin()


def save_model(model):
    f1 = DIRECTORY + '/weights.hdf5'
    f2 = DIRECTORY + '/config.pkl'
    f3 = DIRECTORY + '/dictionary.npz'

    model.model.save_weights(f1)

    config = {'enc_layer': model.enc_layer_output,
              'dec_layer': model.dec_layer_output,
              'seq_len': model.sequence_len}
    pickle.dump(config, open(f2, 'wb'), pickle.HIGHEST_PROTOCOL)

    np.savez(f3, wit=model.word_to_index, itw=model.index_to_word)
    print('Saved model to %s' % DIRECTORY)


def load_model(directory):
    f1 = directory + '/weights.hdf5'
    f2 = directory + '/config.pkl'
    f3 = directory + '/dictionary.npz'

    print('Loading model from %s...' % directory)
    try:
        config = pickle.load(open(f2, 'rb'))

        npz_file = np.load(f3)
        word_to_index, index_to_word = npz_file["wit"].reshape(1)[0], npz_file["itw"]

        print('Done.')
        return LSTMEncDec(None, word_to_index, index_to_word, weight_file=f1,
                          enc_layer_output=config['enc_layer'], dec_layer_output=config['dec_layer'],
                          sequence_len=config['seq_len'])
    except FileNotFoundError:
        print('One or more model files cannot be found. Terminating...')
        exit()
