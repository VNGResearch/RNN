#! /usr/bin/env python

import numpy as np
import nltk
import sys
import operator
import theano
import settings
from gru_theano import GRUTheano
from lstm_theano import LSTMTheano
from model_wrapper import RNNWrapper

SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"


def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, decay=0.9,
                   callback_every=10000, callback=None):
    num_examples_seen = 0
    length = len(y_train)
    for epoch in range(nepoch):
        print("\nStarting epoch %s..." % epoch)
        # For each training example...
        for i in np.random.permutation(length):
            # One SGD step
            print("\rSGD step %s/%s" % (num_examples_seen, length*nepoch), sep=' ', end='', flush=True)
            sys.stdout.flush()
            model.sgd_step(X_train[i], y_train[i], learning_rate, decay)
            num_examples_seen += 1
            # Optionally do callback
            if callback and callback_every and num_examples_seen % callback_every == 0:
                callback(model, num_examples_seen)
    return model


def save_model_parameters_theano(model, outfile):
    np.savez(outfile,
             E=model.E.get_value(),
             U=model.U.get_value(),
             W=model.W.get_value(),
             V=model.V.get_value(),
             b=model.b.get_value(),
             c=model.c.get_value())
    print("Saved model parameters to %s." % outfile)


def load_model_parameters_theano(path, modelClass=GRUTheano):
    npzfile = np.load(path)
    E, U, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
    hidden_dim, word_dim = E.shape[0], E.shape[1]
    print("Building model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim))
    sys.stdout.flush()
    model = modelClass(word_dim, hidden_dim=hidden_dim)
    model.E.set_value(E.astype(theano.config.floatX))
    model.U.set_value(U.astype(theano.config.floatX))
    model.W.set_value(W.astype(theano.config.floatX))
    model.V.set_value(V.astype(theano.config.floatX))
    model.b.set_value(b.astype(theano.config.floatX))
    model.c.set_value(c.astype(theano.config.floatX))
    return model


def save_wrapper(wrapper, outfile):
    np.savez(outfile,
             E=wrapper.model.E.get_value(),
             U=wrapper.model.U.get_value(),
             W=wrapper.model.W.get_value(),
             V=wrapper.model.V.get_value(),
             b=wrapper.model.b.get_value(),
             c=wrapper.model.c.get_value(),
             wit=wrapper.word_to_index,
             itw=wrapper.index_to_word
             )
    print("Saved wrapper to %s." % outfile)


def load_wrapper(path, modelClass=LSTMTheano):
    npzfile = np.load(path)
    E, U, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
    wit, itw = npzfile["wit"].reshape(1)[0], npzfile["itw"]
    hidden_dim, word_dim = E.shape[0], E.shape[1]
    print("Building model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim))
    sys.stdout.flush()
    model = modelClass(word_dim, hidden_dim=hidden_dim)
    print('Creating wrapper...')
    wrapper = RNNWrapper(model, wit, itw)
    return wrapper


def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to check.
    model_parameters = ['E', 'U', 'W', 'b', 'V', 'c']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x], [y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x], [y])
            estimated_gradient = (gradplus - gradminus) / (2 * h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                print("+h Loss: %f" % gradplus)
                print("-h Loss: %f" % gradminus)
                print("Estimated_gradient: %f" % estimated_gradient)
                print("Backpropagation gradient: %f" % backprop_gradient)
                print("Relative Error: %f" % relative_error)
                return
            it.iternext()
        print("Gradient check for parameter %s passed." % (pname))


def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    print(" ".join(sentence_str))
    sys.stdout.flush()
    return sentence_str


def generate_sentence(model, index_to_word, word_to_index, min_length=5):
    # We start the sentence with the start token
    new_sentence = [word_to_index[SENTENCE_START_TOKEN]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        next_word_probs = model.predict(new_sentence)[-1]
        samples = np.random.multinomial(1, next_word_probs)
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        # Sometimes we get stuck if the sentence becomes too long, e.g. "........" :(
        # And: We don't want sentences with UNKNOWN_TOKEN's
        if len(new_sentence) > 100 or sampled_word == word_to_index[UNKNOWN_TOKEN]:
            return None
    if len(new_sentence) < min_length:
        return None
    return new_sentence


def generate_sentences(model, n, index_to_word, word_to_index):
    for i in range(n):
        sent = None
        while not sent:
            try:
                sent = generate_sentence(model, index_to_word, word_to_index)
            except ValueError:
                pass
        print_sentence(sent, index_to_word)


def string_to_index(string, word_to_index):
    tokens = nltk.word_tokenize(string)
    for i in range(len(tokens)):
        tokens[i] = tokens[i] if tokens[i] in word_to_index else UNKNOWN_TOKEN
    indexes = [word_to_index[w] for w in tokens]
    return indexes


def generate_response(question, model, index_to_word, word_to_index, min_length=0):
    q = [word_to_index[SENTENCE_START_TOKEN]]
    q.extend(string_to_index(question.lower(), word_to_index))
    first_word_probs = model.predict(q)[-1]
    first_word = word_to_index[UNKNOWN_TOKEN]
    while first_word == word_to_index[UNKNOWN_TOKEN]:
        first_samples = np.random.multinomial(1, first_word_probs)
        #first_samples = first_word_probs
        first_word = np.argmax(first_samples)
    response = [word_to_index[SENTENCE_START_TOKEN], first_word]
    exchange = list(q)
    exchange.extend(response[1:])
    while not response[-1] == word_to_index[SENTENCE_END_TOKEN]:
        next_word_probs = model.predict(exchange)[-1]
        samples = np.random.multinomial(1, next_word_probs)
        #samples = next_word_probs
        next_word = np.argmax(samples[:-1])
        response.append(next_word)
        exchange.append(next_word)
        if len(response) > 100:
            return response
    if len(response) < min_length:
        return generate_response(question, model, index_to_word, word_to_index, min_length)
    return response
