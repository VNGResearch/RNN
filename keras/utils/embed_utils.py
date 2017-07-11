import array
import numpy as np


class EmbeddingLoader(object):
    """
    A partial clone of the Glove class from glove-python (https://github.com/maciejkula/glove-python)
    for loading Glove embeddings text files.
    """
    def __init__(self):
        self.word_vectors = None
        self.dictionary = None
        self.inverse_dictionary = None

    @classmethod
    def load_stanford_glove(cls, filename):
        return EmbeddingLoader._load_txt_embedding_(filename)

    @classmethod
    def load_fb_fasttext(cls, filename):
        return EmbeddingLoader._load_txt_embedding_(filename, skiplines=1)

    @classmethod
    def _load_txt_embedding_(cls, filename, skiplines=0):
        dct = {}
        vectors = array.array('d')

        # Read in the data.
        entries = []
        with open(filename, 'rt', encoding='utf-8') as savefile:
            lines = savefile.readlines()
            for i, line in enumerate(lines[skiplines:]):
                tokens = line[:-1].split(' ')
                word = tokens[0]
                entries = tokens[1:]
                if '' in tokens:
                    tokens.remove('')

                dct[word] = i
                try:
                    vectors.extend(float(x) for x in entries)
                except ValueError:
                    pass

        # Infer word vectors dimensions.
        no_components = len(entries)
        no_vectors = len(dct)

        # Set up the model instance.
        instance = EmbeddingLoader()
        instance.word_vectors = (np.array(vectors)
                                 .reshape(no_vectors,
                                          no_components))
        instance.add_dictionary(dct)

        return instance

    def add_dictionary(self, dictionary):
        """
        Supply a word-id dictionary to allow similarity queries.
        """
        if self.word_vectors is None:
            raise Exception('Model must be fit before adding a dictionary')

        if len(dictionary) > self.word_vectors.shape[0]:
            raise Exception('Dictionary length must be smaller '
                            'or equal to the number of word vectors')

        self.dictionary = dictionary
        if hasattr(self.dictionary, 'iteritems'):
            # Python 2 compat
            items_iterator = self.dictionary.iteritems()
        else:
            items_iterator = self.dictionary.items()

        self.inverse_dictionary = {v: k for k, v in items_iterator}
