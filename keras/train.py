import gc
from settings_alt import *
from data_utils import *
from lstm_enc_dec import LSTMEncDec2

# Set up the session's directory for storing the model and log file
if not os.path.exists(ALT_DIRECTORY):
    os.makedirs(ALT_DIRECTORY)

# Get training and test data
X, y, word_to_index, index_to_word, word_vec, samples, output_mask = load_data_shakespeare(
    vocabulary_size=VOCABULARY_SIZE,
    sample_size=None,
    sequence_len=SEQUENCE_LENGTH,
    vec_labels=False)
X_val = X[:1500]
X_train = X[1500:]
y_val = y[:1500]
y_train = y[1500:]
X = y = None
gc.collect()

# Get queries
with open(QUERY_FILE, 'rt') as f:
    queries = [q.rstrip() for q in f.readlines()]
    f.close()
queries.extend(samples)

# Initialize model
print('Creating model...')
model = LSTMEncDec2(word_vec, word_to_index, index_to_word, enc_layer_output=ENCODER_OUTPUTS,
                    dec_layer_output=DECODER_OUTPUTS, learning_rate=LEARNING_RATE, sequence_len=SEQUENCE_LENGTH,
                    directory=ALT_DIRECTORY, decoder_type=1)

# Start training
print('Starting training...')
model.train(X_train, y_train, N_EPOCH, batch_size=BATCH_SIZE, queries=queries, Xval=X_val, yval=y_val,
            train_mask=output_mask[1500:], val_mask=output_mask[:1500])
