
class RNNWrapper(object):
    def __init__(self, model, word_to_index, index_to_word):
        self.model = model
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
