LEARNING_RATE = 0.001
VOCABULARY_SIZE = 40000
N_EPOCH = 200
QUERY_FILE = './queries.txt'
SEQUENCE_LENGTH = 500
ENCODER_OUTPUTS = (1500,)
DECODER_OUTPUTS = (1500,)
DATA_SIZE = None  # Number of conversations to extract from yahoo, cornell or southpark dataset
DOC_COUNT = None  # Number of documents to extract from opensub or shakespeare dataset
VAL_SPLIt = 100  # Number of validation samples to be taken from training set
BATCH_SIZE = 20
DATASET = 'cornell'
# Structural settings
OUTPUT_TYPE = 1
DECODER_TYPE = 1
