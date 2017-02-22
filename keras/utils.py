import numpy as np
import datetime
import pickle
from lstm_keras import *

SENTENCE_END_TOKEN = 'SENTENCE_END_TOKEN'
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
MASK_TOKEN = 'MASK_TOKEN'
DIRECTORY = './models/LSTM_%s' % datetime.date.today().isoformat()
ALT_DIRECTORY = './models/LSTM2_%s' % datetime.date.today().isoformat()


def log(string='', out=True, alt=False):
    if not alt:
        f = open(DIRECTORY + '/log.txt', mode='at')
    else:
        f = open(ALT_DIRECTORY + '/log.txt', mode='at')
    if out:
        print(string)
    print(string, file=f)
    f.close()


def nearest_vector(array, value):
    return array[nearest_vector_index(array, value)]


def nearest_vector_index(array, value):
    return np.sum(np.abs(array-value), axis=1).argmin()


def save_model(model, alt=False):
    if alt:
        directory = ALT_DIRECTORY
    else:
        directory = DIRECTORY

    f1 = directory + '/weights.hdf5'
    f2 = directory + '/config.pkl'
    f3 = directory + '/dictionary.npz'

    model.model.save_weights(f1)

    word_vec = model.embed.get_weights()[0]
    config = {'enc_layer': model.enc_layer_output,
              'dec_layer': model.dec_layer_output,
              'seq_len': model.sequence_len,
              'word_vec_dim': np.shape(word_vec)}
    pickle.dump(config, open(f2, 'wb'), pickle.HIGHEST_PROTOCOL)

    np.savez(f3, wit=model.word_to_index, itw=model.index_to_word,
             wv=word_vec)
    print('Saved model to %s' % directory)


def load_model(directory):
    f1 = directory + '/weights.hdf5'
    f2 = directory + '/config.pkl'
    f3 = directory + '/dictionary.npz'

    print('Loading model from %s...' % directory)
    try:
        config = pickle.load(open(f2, 'rb'))

        npz_file = np.load(f3)
        word_to_index, index_to_word, word_vec = npz_file["wit"].reshape(1)[0], npz_file["itw"], npz_file["wv"].reshape(config['word_vec_dim'])

        print('Done.')
        return LSTMEncDec(word_vec, word_to_index, index_to_word, weight_file=f1,
                          enc_layer_output=config['enc_layer'], dec_layer_output=config['dec_layer'],
                          sequence_len=config['seq_len'])
    except FileNotFoundError:
        print('One or more model files cannot be found. Terminating...')


def generate_batch(Xtrain, ytrain, nb_class, total_len, batch_size=10):
    while True:
        for i in range(0, total_len, batch_size):
            yt = to_hot_coded(ytrain[i:i + batch_size], nb_class)
            Xt = Xtrain[i:i + batch_size]
            yield (Xt, yt)


def to_hot_coded(y, nb_classes):
    yt = np.zeros((np.size(y, 0), np.size(y, 1), nb_classes))
    for i in range(np.size(y, 0)):
        for j in range(np.size(y, 1)):
            yt[i][j][int(y[i][j])] = 1

    return yt
