import gc
import os
import sys

from utils.data_utils import get_loader
from utils.commons import *
from lstm.enc_dec import LSTMEncDec
from settings_enc_dec import *

# Alternate name for model passed as argument
if len(sys.argv) > 1:
    DIRECTORY = 'models/' + sys.argv[1]

# Set up the session's directory for storing the model and log file
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

loader, switch = get_loader(DATASET)
if switch:
    DOC_COUNT = DATA_SIZE

# Get training and test data
X, y, word_to_index, index_to_word, word_vec, samples, output_mask = loader(vocabulary_size=VOCABULARY_SIZE,
                                                                            sample_size=DOC_COUNT,
                                                                            sequence_len=SEQUENCE_LENGTH,
                                                                            vec_labels=(OUTPUT_TYPE == 0))
X_val = X[:VAL_SPLIT]
X_train = X[VAL_SPLIT:]
y_val = y[:VAL_SPLIT]
y_train = y[VAL_SPLIT:]
X = y = None
gc.collect()

# Get queries
with open(QUERY_FILE, 'rt') as f:
    queries = [q.rstrip() for q in f.readlines()]
    f.close()
queries.extend(samples)

# Initialize model
print('Creating model...')
model = LSTMEncDec(word_vec, word_to_index, index_to_word, enc_layer_output=ENCODER_OUTPUTS,
                   dec_layer_output=DECODER_OUTPUTS, learning_rate=LEARNING_RATE, sequence_len=SEQUENCE_LENGTH,
                   directory=DIRECTORY, decoder_type=DECODER_TYPE, out_type=OUTPUT_TYPE)

# Start training
print('Starting training...')
model.train(X_train, y_train, N_EPOCH, batch_size=BATCH_SIZE, queries=queries, Xval=X_val, yval=y_val,
            train_mask=output_mask[VAL_SPLIT:], val_mask=output_mask[:VAL_SPLIT])
