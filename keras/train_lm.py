import os
import logging

from lstm.lang_model import LSTMLangModel
from settings_lm import *
from utils import *

__LOGGER = logging.getLogger()
__LOGGER.level = logging.INFO

# Set up the session's directory for storing the model and log file
if not os.path.exists(LM_DIRECTORY):
    os.makedirs(LM_DIRECTORY)

loader, switch = get_loader(DATASET)
if switch:
    DOC_COUNT = DATA_SIZE

# Get training and test data
word_vec, word_to_index, index_to_word = load_embedding(VOCABULARY_SIZE,
                                                        embed_path=EMBEDDING_PATH,
                                                        embed_type=EMBEDDING_TYPE)
X, y, samples, output_mask = loader(word_vec, word_to_index, index_to_word,
                                    sample_size=DOC_COUNT,
                                    sequence_len=SEQUENCE_LENGTH,
                                    vec_labels=False)
del y, output_mask
y, output_mask = generate_lm_labels(X, word_to_index)

X_val = X[:VAL_SPLIT]
X_train = X[VAL_SPLIT:]
y_val = y[:VAL_SPLIT]
y_train = y[VAL_SPLIT:]
del X, y

# Initialize model
logging.info('Creating model...')
model = LSTMLangModel(word_vec, word_to_index, index_to_word, learning_rate=LEARNING_RATE, sequence_len=SEQUENCE_LENGTH,
                      directory=LM_DIRECTORY, outputs=OUTPUTS, dropout=DROPOUT)

# Start training
logging.info('Starting training...')
model.train(X_train, y_train, N_EPOCH, batch_size=BATCH_SIZE, Xval=X_val, yval=y_val,
            train_mask=output_mask[VAL_SPLIT:], val_mask=output_mask[:VAL_SPLIT])
