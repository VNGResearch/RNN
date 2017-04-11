LEARNING_RATE = 0.002
VOCABULARY_SIZE = 1000
N_EPOCH = 1
QUERY_FILE = './queries.txt'
SEQUENCE_LENGTH = 45
ENCODER_OUTPUTS = (1024,)
DECODER_OUTPUTS = (1024, 512,)
DATA_SIZE = 200  # Number of conversations to extract from yahoo dataset
DOC_COUNT = 1  # Number of documents to extract from opensub or shakespeare dataset
VAL_SPLIt = 10  # Number of validation samples
BATCH_SIZE = 15

# Structural settings
OUTPUT_TYPE = 1
DECODER_TYPE = 0
