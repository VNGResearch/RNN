import os
import keras.utils.np_utils as knp
from settings_alt import *
from data_utils import *
from lstm_keras import LSTMEncDec2

# Set up the session's directory for storing the model and log file
if not os.path.exists(ALT_DIRECTORY):
    os.makedirs(ALT_DIRECTORY)

# Get training and test data
X_train, y_train, word_to_index, index_to_word, word_vec = load_data_opensub(vocabulary_size=VOCABULARY_SIZE,
                                                                             sample_size=DOC_COUNT,
                                                                             sequence_len=SEQUENCE_LENGTH,
                                                                             vec_labels=False)

# Get queries
with open(QUERY_FILE, 'rt') as f:
    queries = f.readlines()
    f.close()

# Initialize model
print('Creating model...')
model = LSTMEncDec2(word_vec, word_to_index, index_to_word, enc_layer_output=ENCODER_OUTPUTS,
                    dec_layer_output=DECODER_OUTPUTS, learning_rate=LEARNING_RATE, sequence_len=SEQUENCE_LENGTH)

# Start training
print('Starting training...')
model.train(X_train, y_train, N_EPOCH, BATCH_SIZE, queries=queries)
