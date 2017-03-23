import json
import nltk
import os
import random
import untangle
import numpy as np
from glove import Glove
from utils import *


def load_embedding(vocabulary_size):
    print("Loading word embedding...")
    embed = Glove.load_stanford('data/glove.6B.100d.txt')
    embed_layer = np.asarray(embed.word_vectors[:vocabulary_size, :], dtype=np.float32)
    index_to_word = list(embed.inverse_dictionary.values())
    index_to_word = index_to_word[:vocabulary_size - 3]
    index_to_word.insert(0, MASK_TOKEN)
    index_to_word.append(SENTENCE_END_TOKEN)
    index_to_word.append(UNKNOWN_TOKEN)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    word_dim = np.size(embed_layer, 1)
    # Vector for the MASK token
    embed_layer = np.vstack((np.zeros((1, word_dim), dtype=np.float32), embed_layer))
    # TODO Embed meaning for SENTENCE_END_TOKEN
    embed_layer = np.vstack((embed_layer, np.asarray(np.random.uniform(15.0, 30.0, (1, word_dim)), dtype=np.float32)))
    # Random vector for UNKNOWN_TOKEN, placed intentionally far away from vocabulary words
    embed_layer = np.vstack((embed_layer, np.asarray(np.random.uniform(50.0, 80.0, (1, word_dim)), dtype=np.float32)))

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
        print('<<%s>>' % fn)
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

    output_mask = np.ones((len(raw_y), sequence_len), dtype=np.float32)
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
                if raw_y[i][j] == SENTENCE_END_TOKEN:
                    # output_mask[i][j] = 1.5
                    for k in range(j+1,sequence_len):
                        output_mask[i][k] = 0

    return X_train, y_train, word_to_index, index_to_word, embed_layer, samples, output_mask


def load_data_shakespeare(path='./data/shakespeare', vocabulary_size=2000, sample_size=None, sequence_len=2000, vec_labels=True):
    def load_play(file_path):
        play = untangle.parse(file_path)
        acts = play.PLAY.ACT
        q, a = [], []

        for act in acts:
            scenes = act.SCENE
            for scene in scenes:
                s_q, s_a = [], []
                speeches = scene.SPEECH
                for speech in speeches:
                    lines = speech.LINE
                    line_str = []
                    for line in lines:
                        line_st = line.cdata
                        if len(line_st) == 0:
                            continue
                        if line_st[-1].isalpha():
                            line_st += '.'
                        speech_tokens = nltk.word_tokenize(line_st.lower())
                        s_q.append(speech_tokens)
                        if len(s_q) > 1:
                            s_a.append(speech_tokens + [SENTENCE_END_TOKEN])
                        # line_str.append(line_st)
                    # speech_str = ' '.join(line_str)
                    # speech_tokens = nltk.word_tokenize(speech_str.lower())
                    # s_q.append(speech_tokens)
                    # if len(s_q) > 1:
                    #     s_a.append(speech_tokens)
                s_q = s_q[:-1]
                q.extend(s_q)
                a.extend(s_a)
        assert len(q) == len(a)
        return q, a

    print('Reading XML files...')
    raw_x, raw_y = [], []
    fl = os.listdir(path)

    if sample_size is not None:
        np.random.shuffle(fl)
        fl = fl[:sample_size]

    print("Using vocabulary size %d." % vocabulary_size)
    embed_layer, word_to_index, index_to_word = load_embedding(vocabulary_size)

    print('Tokenizing...')
    for fn in fl:
        if not fn.endswith('.xml'):
            continue
        print('<<%s>>' % fn)
        rx, ry = load_play(path + '/' + fn)
        raw_x.extend(rx)
        raw_y.extend(ry)

    samples = []
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

    output_mask = np.ones((len(raw_y), sequence_len), dtype=np.float32)
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
                if raw_y[i][j] == SENTENCE_END_TOKEN:
                    # output_mask[i][j] = 1.5
                    for k in range(j + 1, sequence_len):
                        output_mask[i][k] = 0

    return X_train, y_train, word_to_index, index_to_word, embed_layer, samples, output_mask


def remove_symbols(sentence):
    return sentence
