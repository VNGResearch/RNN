import json
import nltk
import os
import random
import numpy as np
from glove import Glove
from utils import *


def load_embedding(vocabulary_size):
    print("Loading word embedding...")
    embed = Glove.load_stanford('data/glove.6B.100d.txt')
    embed_layer = np.asarray(embed.word_vectors[:vocabulary_size, :], dtype=np.float32)
    index_to_word = list(embed.inverse_dictionary.values())
    index_to_word.insert(0, MASK_TOKEN)
    index_to_word = index_to_word[:vocabulary_size]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    word_count = len(index_to_word)
    index_to_word.append(SENTENCE_END_TOKEN)
    word_to_index[SENTENCE_END_TOKEN] = word_count
    index_to_word.append(UNKNOWN_TOKEN)
    word_to_index[UNKNOWN_TOKEN] = word_count + 1

    word_dim = np.size(embed_layer, 1)
    # Vector for the MASK token
    embed_layer = np.vstack((np.zeros((1, word_dim), dtype=np.float32), embed_layer))
    # TODO Embed meaning for SENTENCE_END_TOKEN
    embed_layer = np.vstack((embed_layer, np.asarray(np.random.uniform(-10.0, 10.0, (1, word_dim)), dtype=np.float32)))
    # Random vector for UNKNOWN_TOKEN, placed intentionally far away from vocabulary words
    embed_layer = np.vstack((embed_layer, np.asarray(np.random.uniform(20.0, 50.0, (1, word_dim)), dtype=np.float32)))

    return embed_layer, word_to_index, index_to_word


def load_data_yahoo(filename="data/nfL6.json", vocabulary_size=2000, sample_size=None, sequence_len=2000, vec_labels=True):
    print("Reading JSON file (%s) ..." % filename)
    questions = []
    answers = []
    with open(filename, 'r') as f:
        data = json.load(f)
        if sample_size is not None:
            data = random.sample(data, sample_size)
        for qa in data:
            questions.append("%s" % qa['question'].lower())
            answers.append("%s %s" % (qa['answer'].lower(), SENTENCE_END_TOKEN))

    print("Tokenizing...")
    tokenized_questions = [nltk.word_tokenize(sent) for sent in questions]
    tokenized_answers = [nltk.word_tokenize(sent) for sent in answers]
    print("Parsed %d exchanges." % (len(tokenized_questions)))

    print("Using vocabulary size %d." % vocabulary_size)
    embed_layer, word_to_index, index_to_word = load_embedding(vocabulary_size)

    # Replace all words not in our vocabulary with the unknown token
    # Keep track of the unknown token ratio
    unk_count = 0.0
    total = 0.0
    for i, sent in enumerate(tokenized_questions):
        idx = 0
        for w in sent:
            if w in word_to_index:
                nw = w
            else:
                nw = UNKNOWN_TOKEN
                unk_count += 1.0
            total += 1.0
            tokenized_questions[i][idx] = nw
            idx += 1
    for i, sent in enumerate(tokenized_answers):
        idx = 0
        for w in sent:
            if w in word_to_index:
                nw = w
            else:
                nw = UNKNOWN_TOKEN
                unk_count += 1.0
            total += 1.0
            tokenized_answers[i][idx] = nw
            idx += 1
    print("%s unknown tokens / %s tokens " % (int(unk_count), int(total)))
    print("Unknown token ratio: %s %%" % (unk_count * 100 / total))

    # Create the training data
    print('Generating data...')
    X_train = np.zeros((len(tokenized_questions), sequence_len), dtype=np.int32)
    for i in range(len(tokenized_questions)):
        for j in range(len(tokenized_questions[i])):
            X_train[i][j] = word_to_index[tokenized_questions[i][j]]

    if vec_labels:
        y_train = np.zeros((len(tokenized_answers), sequence_len, np.size(embed_layer, 1)), dtype=np.float32)
        for i in range(len(tokenized_answers)):
            for j in range(len(tokenized_answers[i])):
                y_train[i][j] = embed_layer[word_to_index[tokenized_answers[i][j]]]
    else:
        y_train = np.zeros((len(tokenized_answers), sequence_len), dtype=np.float32)
        for i in range(len(tokenized_answers)):
            for j in range(len(tokenized_answers[i])):
                p = word_to_index[tokenized_answers[i][j]]
                y_train[i][j] = p

    return X_train, y_train, word_to_index, index_to_word, embed_layer


def load_data_opensub(path='./data/opensub', vocabulary_size=2000, sample_size=None, sequence_len=2000, vec_labels=True):
    print('Reading TXT files...')
    raw_x, raw_y = [], []
    fl = os.listdir(path)
    if sample_size is not None:
        np.random.shuffle(fl)
        fl = fl[:sample_size]

    print("Using vocabulary size %d." % vocabulary_size)
    embed_layer, word_to_index, index_to_word = load_embedding(vocabulary_size)

    samples = []
    print('Tokenizing...')
    for fn in fl:
        f = open(path + '/' + fn, 'rt')
        lines = f.readlines()
        for i, l in enumerate(lines[:-1]):
            x = random.uniform(0, 1)
            if x > 0.95:
                samples.append(l.rstrip().lower())

            l1 = nltk.word_tokenize(l.rstrip().lower())[:sequence_len]
            l2 = nltk.word_tokenize(lines[i+1].rstrip().lower())[:sequence_len-1]
            l2.append(SENTENCE_END_TOKEN)
            raw_x.append(l1)
            raw_y.append(l2)
    print("Parsed %s exchanges." % (len(raw_x)))

    unk_count = 0.0
    total = 0.0
    for i, sent in enumerate(raw_x):
        idx = 0
        for w in sent:
            if w in word_to_index:
                nw = w
            else:
                nw = UNKNOWN_TOKEN
                unk_count += 1.0
            total += 1.0
            raw_x[i][idx] = nw
            idx += 1
    for i, sent in enumerate(raw_y):
        idx = 0
        for w in sent:
            if w in word_to_index:
                nw = w
            else:
                nw = UNKNOWN_TOKEN
                unk_count += 1.0
            total += 1.0
            raw_y[i][idx] = nw
            idx += 1
    print("%s unknown tokens / %s tokens " % (int(unk_count), int(total)))
    print("Unknown token ratio: %s %%" % (unk_count * 100 / total))

    print('Generating data...')
    X_train = np.zeros((len(raw_x), sequence_len), dtype=np.int32)
    for i in range(len(raw_x)):
        for j in range(len(raw_x[i])):
            X_train[i][j] = word_to_index[raw_x[i][j]]

    if vec_labels:
        y_train = np.zeros((len(raw_y), sequence_len, np.size(embed_layer, 1)), dtype=np.float32)
        for i in range(len(raw_y)):
            for j in range(len(raw_y[i])):
                y_train[i][j] = embed_layer[word_to_index[raw_y[i][j]]]
    else:
        y_train = np.zeros((len(raw_y), sequence_len), dtype=np.float32)
        for i in range(len(raw_y)):
            for j in range(len(raw_y[i])):
                p = word_to_index[raw_y[i][j]]
                y_train[i][j] = p

    return X_train, y_train, word_to_index, index_to_word, embed_layer, samples
