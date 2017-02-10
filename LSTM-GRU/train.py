#! /usr/bin/env python

import os
import time
import settings
from utils import *
from data_utils import *
from datetime import datetime
from lstm_theano import LSTMTheano
from model_wrapper import RNNWrapper

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", settings.LEARNING_RATE))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", settings.VOCABULARY_SIZE))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", settings.HIDDEN_DIM))
NEPOCH = int(os.environ.get("NEPOCH", settings.NEPOCH))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE", settings.MODEL_OUTPUT_FILE)
WRAPPER_OUTPUT_FILE = os.environ.get("WRAPPER_OUTPUT_FILE", settings.WRAPPER_OUTPUT_FILE)
QUERY_FILE = os.environ.get("QUERY_FILE", settings.QUERY_FILE)
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", settings.PRINT_EVERY))
EMBEDDING = bool(os.environ.get("EMBEDDING", settings.EMBEDDING))
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", settings.SAMPLE_SIZE))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))


def log(string=''):
    f = open('.log', mode='at')
    print(string, file=f)
    f.close()


def sgd_response_callback(model, num_examples_seen):
    dt = datetime.now().isoformat()
    loss = model.calculate_loss(x_train[:10000], y_train[:10000])
    print("\n%s (%d)" % (dt, num_examples_seen))
    print("--------------------------------------------------")
    print("Loss: %f" % loss)
    for i in range(5):
        question = input('Question %s: ' % i)
        response = generate_response(question, model, index_to_word, word_to_index)
        print_sentence(response, index_to_word)
    save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
    print("\n")


def sgd_auto_callback(model, num_examples_seen):
    dt = datetime.now().isoformat()
    loss = model.calculate_loss(x_train, y_train)
    print("\n%s (%d)" % (dt, num_examples_seen))
    print("--------------------------------------------------")
    print("Loss: %f" % loss)
    log("\n(%d)" % num_examples_seen)
    log("--------------------------------------------------")
    log("Loss: %f" % loss)

    try:
        with open(QUERY_FILE, 'rt') as f:
            queries = f.readlines()

        for question in queries:
            print('Q: ' + question, end='')
            log('Q: ' + question)
            response = generate_response(question, model, index_to_word, word_to_index)
            sentence = print_sentence(response, index_to_word)
            log(' '.join(sentence) + '\n')
            print()
    except IOError:
        print('Warning: No input file found')
        pass
    wrapper = RNNWrapper(model, word_to_index, index_to_word)
    save_wrapper(wrapper, WRAPPER_OUTPUT_FILE)
    print("\n")


# Load data
x_train, y_train, word_to_index, index_to_word, embedder = \
    load_data_yahoo(vocabulary_size=VOCABULARY_SIZE, embedding=EMBEDDING, sample_size=SAMPLE_SIZE)

# Build model
print('Building model...')
if embedder is not None:
    HIDDEN_DIM = np.size(embedder, 1)
    VOCABULARY_SIZE = np.size(embedder, 0)
model = LSTMTheano(word_dim=VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1, embedder=embedder)

# Set up output for model
if not MODEL_OUTPUT_FILE:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    MODEL_OUTPUT_FILE = "./models/LSTM-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
if not WRAPPER_OUTPUT_FILE:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    WRAPPER_OUTPUT_FILE = "./models/LSTM-%s-%s-%s.wrp" % (ts, VOCABULARY_SIZE, HIDDEN_DIM)

# Set up log file
dt = datetime.now().isoformat()
log('%s - HIDDEN_DIM=%s - VOCABULARY_SIZE=%s' % (dt, HIDDEN_DIM, VOCABULARY_SIZE))

# Print SGD step time
t1 = time.time()
model.sgd_step(x_train[10], y_train[10], LEARNING_RATE)
t2 = time.time()
print("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))

# Start training
print("Starting training...")
train_with_sgd(model, x_train, y_train, learning_rate=LEARNING_RATE,
               nepoch=NEPOCH, decay=0.9, callback_every=PRINT_EVERY, callback=sgd_auto_callback)
