LEARNING_RATE = 0.0003
VOCABULARY_SIZE = 20000
N_EPOCH = 200
QUERY_FILE = './queries.txt'
SEQUENCE_LENGTH = 45
ENCODER_OUTPUTS = (1000,)
DECODER_OUTPUTS = (1000,)
DATA_SIZE = 50000  # Number of conversations to extract from yahoo, cornell or southpark dataset
DOC_COUNT = None  # Number of documents to extract from opensub or shakespeare dataset
VAL_SPLIT = 100  # Number of validation samples to be taken from training set
BATCH_SIZE = 20
DATASET = 'cornell'
# Structural settings
OUTPUT_TYPE = 1
DECODER_TYPE = 2
