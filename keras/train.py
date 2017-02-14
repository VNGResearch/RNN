import numpy as np
import os
from utils import *
from settings import *
from data_utils import *
from lstm_keras import LSTMEncDec

# Set up the session's directory for storing the model and log file
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

# Get training and test data
X_train, y_train, word_to_index, index_to_word, embed_layer = load_data_yahoo(vocabulary_size=VOCABULARY_SIZE, sample_size=DATA_SIZE)

