from settings_lm import *
from lstm_lang_model import LSTMLangModel
from data_utils import *

import gc


LM_DIRECTORY = './lang_models/LSTM_%s' % datetime.date.today().isoformat()

# Set up the session's directory for storing the model and log file
if not os.path.exists(LM_DIRECTORY):
    os.makedirs(LM_DIRECTORY)

loader, switch = get_loader(DATASET)
if switch:
    DOC_COUNT = DATA_SIZE

# Get training and test data
X, _y, word_to_index, index_to_word, word_vec, samples, _output_mask = loader(vocabulary_size=VOCABULARY_SIZE,
                                                                              sample_size=DOC_COUNT,
                                                                              sequence_len=SEQUENCE_LENGTH,
                                                                              vec_labels=False)
y, output_mask = generate_lm_labels(X, word_to_index)

X_val = X[:VAL_SPLIt]
X_train = X[VAL_SPLIt:]
y_val = y[:VAL_SPLIt]
y_train = y[VAL_SPLIt:]
X = y = _y = _output_mask = None
gc.collect()

# Initialize model
print('Creating model...')
model = LSTMLangModel(word_vec, word_to_index, index_to_word, learning_rate=LEARNING_RATE, sequence_len=SEQUENCE_LENGTH,
                      directory=LM_DIRECTORY, outputs=OUTPUTS)

# Start training
print('Starting training...')
model.train(X_train, y_train, N_EPOCH, batch_size=BATCH_SIZE, Xval=X_val, yval=y_val,
            train_mask=output_mask[VAL_SPLIt:], val_mask=output_mask[:VAL_SPLIt])
