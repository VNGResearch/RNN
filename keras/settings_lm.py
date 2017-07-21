LEARNING_RATE = 0.0005
VOCABULARY_SIZE = 50000
N_EPOCH = 100
SEQUENCE_LENGTH = 50
OUTPUTS = (2000, 1000,)
DATA_SIZE = None  # Number of conversations to extract from yahoo, cornell or southpark dataset
DOC_COUNT = None  # Number of documents to extract from opensub or shakespeare dataset
VAL_SPLIT = 200  # Number of validation samples to be taken from training set
BATCH_SIZE = 100
DROPOUT = 0.5
DATASET = 'vnnews'

# Word Embdedding settings
EMBEDDING_TYPE = 'fasttext'
EMBEDDING_PATH = 'data/wiki.vi.vec'
# EMBEDDING_TYPE = 'glove'
# EMBEDDING_PATH = 'data/glove.6B.100d.txt'
