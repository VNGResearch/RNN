import numpy as np
import datetime
import pickle
import sys

SENTENCE_END_TOKEN = 'SENTENCE_END_TOKEN'
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
MASK_TOKEN = 'MASK_TOKEN'
DIRECTORY = './models/ED_%s' % datetime.date.today().isoformat()
LM_DIRECTORY = './models/LM_%s' % datetime.date.today().isoformat()


def nearest_vector(array, value):
    return array[nearest_vector_index(array, value)]


def nearest_vector_index(array, value):
    return np.sum(np.abs(array-value), axis=1).argmin()


def save_model(model):
    directory = model.directory

    f1 = directory + '/weights.hdf5'
    f2 = directory + '/config.pkl'
    f3 = directory + '/dictionary.npz'

    model.model.save_weights(f1)

    word_vec = model.embed.get_weights()[0]
    config = {'enc_layer': model.enc_layer_output,
              'dec_layer': model.dec_layer_output,
              'seq_len': model.sequence_len,
              'word_vec_dim': np.shape(word_vec),
              'decoder_type': model.decoder_type,
              'out_type': model.out_type}
    pickle.dump(config, open(f2, 'wb'), pickle.HIGHEST_PROTOCOL)

    np.savez(f3, wit=model.word_to_index, itw=model.index_to_word,
             wv=word_vec)
    print('Saved model to %s' % directory)


def load_model(directory, m_class):
    f1 = directory + '/weights.hdf5'
    f2 = directory + '/config.pkl'
    f3 = directory + '/dictionary.npz'

    print('Loading model from %s...' % directory)
    try:
        config = pickle.load(open(f2, 'rb'))

        npz_file = np.load(f3)
        word_to_index, index_to_word, word_vec = npz_file["wit"].reshape(1)[0], npz_file["itw"], npz_file["wv"].reshape(config['word_vec_dim'])

        print('Done.')
        return m_class(word_vec, word_to_index, index_to_word, weight_file=f1,
                       enc_layer_output=config['enc_layer'], dec_layer_output=config['dec_layer'],
                       sequence_len=config['seq_len'], decoder_type=config.get('decoder_type', 0),
                       out_type=config.get('out_type', 1), directory=directory)
    except FileNotFoundError:
        print('One or more model files cannot be found. Terminating...')
        sys.exit()


def generate_batch(Xtrain, ytrain, word_vec, mask, nb_class, total_len, batch_size=10):
    while True:
        for i in range(0, total_len, batch_size):
            yt = to_hot_coded(ytrain[i:i + batch_size], nb_class)
            Xt = Xtrain[i:i + batch_size]
            msk = mask[i:i + batch_size]
            yield (Xt, yt, msk)


def generate_vector_batch(Xtrain, ytrain, word_vec, mask, nb_class, total_len, batch_size=10):
    while True:
        for i in range(0, total_len, batch_size):
            yt = to_vector(ytrain[i:i + batch_size], word_vec)
            Xt = Xtrain[i:i + batch_size]
            msk = mask[i:i + batch_size]
            yield (Xt, yt, msk)


def to_hot_coded(y, nb_classes):
    yt = np.zeros((np.size(y, 0), np.size(y, 1), nb_classes), dtype=np.float32)
    for i in range(np.size(y, 0)):
        for j in range(np.size(y, 1)):
            yt[i][j][int(y[i][j])] = 1

    return yt


def to_vector(y, word_vec):
    yt = yt = np.zeros((np.size(y, 0), np.size(y, 1), np.size(word_vec, 1)), dtype=np.float32)
    for i in range(np.size(y, 0)):
        for j in range(np.size(y, 1)):
            yt[i][j] = word_vec[int(y[i][j])]

    return yt


def k_largest_idx(array3d, k):
    return np.argsort(array3d)[:, :, ::-1][:, :, :k]
