import nltk
from lstm import utils
from keras.callbacks import *
from keras.layers import Input, Dense, TimeDistributed
from keras.layers.core import RepeatVector
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from recurrentshop import RecurrentContainer, LSTMCell
from lstm.callbacks import EncDecCallback


class LSTMEncDec:
    __LOSS_FUNCS__ = ('mean_squared_error', 'categorical_crossentropy')
    __DECODER_BUILDS__ = None

    def __init__(self, word_vec, word_to_index, index_to_word, weight_file=None, enc_layer_output=(32,),
                 dec_layer_output=(32,), learning_rate=0.001, sequence_len=200, output_len=2000, directory='.',
                 out_type=0, decoder_type=0):
        """
        :param out_type: 0 for word vector output/similarity inference, 1 for softmax word distribution output.
        :param decoder_type: 0 for non-readout LSTM decoder, 1 for recurrentshop's readout decoder, 2 for seq2seq decoder.
        """
        self.__DECODER_BUILDS__ = (self.__build_repeat_decoder__,
                                   self.__build_readout_decoder__,
                                   self.__build_seq2seq_decoder__)
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.sequence_len = sequence_len
        self.output_len = output_len
        self.directory = directory
        self.enc_layer_output = enc_layer_output
        self.dec_layer_output = dec_layer_output
        self.decoder_type = decoder_type
        self.out_type = out_type
        try:
            loss = self.__LOSS_FUNCS__[out_type]
        except IndexError:
            raise ValueError('Invalid output type %s.' % self.out_type)
        self.encoder = Sequential(name='Encoder')
        self.decoder = Sequential(name='Decoder')
        self.embed = None
        self.batch_size = 0

        input_layer, output_layer = self.config_model(word_vec)

        self.model = Model(inputs=[input_layer], outputs=[output_layer])
        if weight_file is not None:
            self.model.load_weights(weight_file)
        self.compile(learning_rate, loss)

    def config_model(self, word_vec):
        """
        Creates the encoder-decoder structure and returns the symbolic input and output
        """
        train_embed = True

        # Configure input layer
        input_layer = Input(shape=(self.sequence_len,), name='Input')

        # Embedding layer should be initialized with a word-vector array and
        # not be trained as the output relies on the same array
        self.embed = Embedding(input_dim=np.size(word_vec, 0), output_dim=np.size(word_vec, 1),
                               weights=[word_vec], trainable=train_embed, mask_zero=True, name='Embed')

        # Configure encoder network with the given output sizes.
        # Embedding for encoder only since decoder receives the question vector.
        self.encoder.add(self.embed)
        for el in self.enc_layer_output[:-1]:
            self.encoder.add(LSTM(el, return_sequences=True, consume_less='mem'))
        self.encoder.add(LSTM(self.enc_layer_output[-1]))  # Final LSTM layer only outputs the last vector
        # Encoder outputs the question vector as a tensor with each time-step output being the final question vector
        question_vec = self.encoder(input_layer)

        # Configure decoder network with the given output sizes.
        # Layer connecting to encoder output
        try:
            self.__DECODER_BUILDS__[self.out_type]()
        except IndexError:
            raise ValueError('Invalid decoder type %s.' % self.decoder_type)

        if self.out_type == 0:
            # Final layer outputting a sequence of word vectors
            self.decoder.add(TimeDistributed(Dense(np.size(word_vec, 1), activation='linear')))
        else:
            # Final layer outputting a sequence of word distribution vectors
            self.decoder.add(TimeDistributed(Dense(len(self.index_to_word), activation='softmax')))
        output_layer = self.decoder(question_vec)

        return input_layer, output_layer

    def __build_repeat_decoder__(self):
        # Repeat the final vector for answer input
        self.decoder.add(RepeatVector(self.output_len, input_shape=(self.enc_layer_output[-1],)))
        self.decoder.add(LSTM(self.dec_layer_output[0], input_shape=(self.output_len, self.enc_layer_output[-1]),
                              name='ConnectorLSTM', return_sequences=True, consume_less='mem'))
        for dl in self.dec_layer_output[1:]:
            self.decoder.add(LSTM(dl, return_sequences=True, consume_less='mem'))

    def __build_readout_decoder__(self):
        self.decoder.add(RepeatVector(self.output_len, input_shape=(
            self.enc_layer_output[-1],)))  # Repeat the final vector for answer input
        # Using recurrentshop's container with readout
        container = RecurrentContainer(readout=True, return_sequences=True,
                                       output_length=self.output_len)
        if len(self.dec_layer_output) > 1:
            container.add(LSTMCell(output_dim=self.dec_layer_output[0],
                                   input_dim=self.enc_layer_output[-1]))
            for dl in self.dec_layer_output[1:-1]:
                container.add(LSTMCell(output_dim=dl))
            container.add(LSTMCell(output_dim=self.enc_layer_output[-1]))
        else:
            container.add(LSTMCell(input_dim=self.enc_layer_output[-1],
                                   output_dim=self.enc_layer_output[-1]))

        if self.enc_layer_output[-1] != self.dec_layer_output[-1]:
            print('WARNING: Overriding final decoder output to %s for readout compatibility' %
                  self.enc_layer_output[-1])
        self.decoder.add(container)

    def __build_seq2seq_decoder__(self):
        # Using recurrentshop's decoder container
        container = RecurrentContainer(return_sequences=True, readout='add',
                                       output_length=self.output_len,
                                       input_shape=(self.enc_layer_output[-1],),
                                       decode=True)
        if len(self.dec_layer_output) > 1:
            container.add(LSTMCell(output_dim=self.dec_layer_output[0],
                                   input_dim=self.enc_layer_output[-1]))
            for dl in self.dec_layer_output[1:-1]:
                container.add(LSTMCell(output_dim=dl))
            container.add(LSTMCell(output_dim=self.enc_layer_output[-1]))
        else:
            container.add(LSTMCell(input_dim=self.enc_layer_output[-1],
                                   output_dim=self.enc_layer_output[-1]))

        if self.enc_layer_output[-1] != self.dec_layer_output[-1]:
            print('WARNING: Overriding final decoder output to %s for readout compatibility' %
                  self.enc_layer_output[-1])
        self.decoder.add(container)

    def compile(self, learning_rate, loss):
        if self.out_type == 0:
            metrics = ['mean_absolute_error']
        else:
            metrics = []
        self.model.compile(optimizer=RMSprop(lr=learning_rate), loss=loss, metrics=metrics,
                           sample_weight_mode='temporal')

    def train(self, Xtrain, ytrain, nb_epoch, Xval=None, yval=None, train_mask=None, val_mask=None, batch_size=10,
              queries=None):
        """
        Uses a generator to decompress labels from integers to hot-coded vectors batch-by-batch to save memory.
        See utils.generate_batch().
        """
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
            self.model.fit_generator(
                generator(Xtrain, ytrain, self.embed.get_weights()[0], train_mask, nb_class, total_len, batch_size),
                steps_per_epoch=total_len / batch_size, workers=1,
                epochs=nb_epoch, callbacks=[callback, logger], verbose=1, max_q_size=1)
        else:
            self.model.fit_generator(
                generator(Xtrain, ytrain, self.embed.get_weights()[0], train_mask, nb_class, total_len, batch_size),
                steps_per_epoch=total_len / batch_size, epochs=nb_epoch, callbacks=[callback, logger],
                verbose=1, max_q_size=1, workers=1, validation_steps=Xval.shape[0] / self.batch_size,
                validation_data=generator(Xval, yval, self.embed.get_weights()[0], val_mask, nb_class, Xval.shape[0],
                                          batch_size))

    def generate_response(self, query):
        """
        Pre-processes a raw query string and return a response string
        """
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

    def log(self, string='', out=True):
        f = open(self.directory + '/log.txt', mode='at')

        if out:
            print(string)
        print(string, file=f)
        f.close()