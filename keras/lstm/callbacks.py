import gc

from keras.callbacks import Callback

from lstm import utils


class EncDecCallback(Callback):
    def __init__(self, enc_dec, queries=[], alt=False):
        self.enc_dec = enc_dec
        self.queries = queries
        self.alt = alt
        super(EncDecCallback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        print()
        utils.save_model(self.enc_dec)
        print('Logging responses')
        self.enc_dec.log('Epoch %s:\n' % epoch, out=False)
        for query in self.queries:
            response, candidates = self.enc_dec.generate_candidates(query)
            self.enc_dec.log('Q: %s\nA: %s\n%s\n' % (query, response, candidates), out=False)
        gc.collect()


class LangModelCallback(Callback):
    def __init__(self, lang_model):
        self.lang_model = lang_model

    def on_epoch_end(self, epoch, logs={}):
        print()
        self.lang_model.save()
