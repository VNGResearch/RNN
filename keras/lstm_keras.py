import numpy as np
import nltk
import utils
from callbacks import EncDecCallback
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.layers import Input
from keras.layers.core import RepeatVector
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop


class LSTMEncDec:
    def __init__(self, word_vec, word_to_index, index_to_word, weight_file=None, enc_layer_output=(32,), dec_layer_output=(32,), learning_rate=0.001, sequence_len=2000):
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.sequence_len = sequence_len
        self.enc_layer_output = enc_layer_output
        self.dec_layer_output = dec_layer_output

        # Embedding layer should be initialized with a word-vector array and not be trained as the output relies on the same array
        if word_vec is not None:
            self.embed = Embedding(input_dim=np.size(word_vec, 0), output_dim=np.size(word_vec, 1),
                                   weights=[word_vec], trainable=False, mask_zero=True, name='Embed')
        else:
            self.embed = Embedding(input_dim=np.size(word_vec, 0), output_dim=np.size(word_vec, 1),
                                   trainable=False, mask_zero=True, name='Embed')

        # Configure input layer
        input_layer = Input(shape=(sequence_len,), name='Input')

        # Configure encoder network with the given output sizes.
        self.encoder = Sequential()
        # Embedding for encoder only since decoder receives the question vector.
        self.encoder.add(self.embed)
        for el in enc_layer_output[:-1]:
            self.encoder.add(LSTM(el, return_sequences=True))
        self.encoder.add(LSTM(enc_layer_output[-1]))  # Final LSTM layer only outputs the last vector
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

        if weight_file is not None:
            self.model.load_weights(weight_file)

        self.model = Model(input=input_layer, output=output_layer)
        self.model.compile(optimizer=RMSprop(learning_rate), loss='mean_squared_error')
        self.model.build(sequence_len)

    def train(self, Xtrain, ytrain, nb_epoch, batch_size=10, queries=None):
        callback = EncDecCallback(self, queries)
        self.model.fit(Xtrain, ytrain, nb_epoch=nb_epoch, batch_size=batch_size, callbacks=[callback], verbose=1)

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
