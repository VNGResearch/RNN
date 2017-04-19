LEARNING_RATE = 0.001
VOCABULARY_SIZE = 50000
N_EPOCH = 150
QUERY_FILE = './queries.txt'
SEQUENCE_LENGTH = 50
ENCODER_OUTPUTS = (1200, 1000, )
DECODER_OUTPUTS = (1000,)
DATA_SIZE = 500  # Number of conversations to extract from yahoo, cornell or southpark dataset
DOC_COUNT = 1  # Number of documents to extract from opensub or shakespeare dataset
VAL_SPLIt = 20  # Number of validation samples to be taken from training set
BATCH_SIZE = 15
DATASET = 'southpark'

# Structural settings
OUTPUT_TYPE = 1
DECODER_TYPE = 1
