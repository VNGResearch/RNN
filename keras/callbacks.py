from keras.callbacks import Callback
import utils
import sys


class EncDecCallback(Callback):
    def __init__(self, enc_dec, queries=[], alt=False):
        self.enc_dec = enc_dec
        self.queries = queries
        self.alt = alt
        super(EncDecCallback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        utils.log('\nEnd of epoch %s --- Loss: %f' % (epoch, logs['loss']))
        utils.save_model(self.enc_dec)
        print('Logging responses')
        for query in self.queries:
            response = self.enc_dec.generate_response(query)
            utils.log('Q: %s\nA: %s' % (query, response), out=False, alt=self.alt)
