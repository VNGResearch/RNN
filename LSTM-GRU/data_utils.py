import csv
import itertools
import operator
import nltk
import json
import random
import numpy as np
from glove import Glove

SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
# TODO Possible QUESTION_TOKEN to signal end of question


def load_embedding(vocabulary_size):
    print("Loading word embedding...")
    embed = Glove.load_stanford('data/glove.6B.50d.txt')
    embed_layer = embed.word_vectors[:vocabulary_size, :]
    index_to_word = list(embed.inverse_dictionary.values())
    index_to_word = index_to_word[:vocabulary_size]
    word_to_index = dict([(w, i) for i, w in embed.inverse_dictionary.items()][:vocabulary_size])

    return embed_layer, word_to_index, index_to_word


def load_data_reddit(filename="data/reddit-comments-2015-08.csv", vocabulary_size=2000, min_sent_characters=0,embedding=True):
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")
    with open(filename, 'rt') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Filter sentences
        sentences = [s for s in sentences if len(s) >= min_sent_characters]
        sentences = [s for s in sentences if "http" not in s]
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    print("Tokenizing...")
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    embed_layer = None
    if not embedding:
        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size - 2]
        print("Using vocabulary size %d." % vocabulary_size)
        print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
        index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in sorted_vocab]
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    else:
        print("Using vocabulary size %d." % vocabulary_size)
        embed_layer, word_to_index, index_to_word = load_embedding(vocabulary_size)

        # Add special tokens to vocab
        word_count = len(index_to_word)
        index_to_word.append(SENTENCE_START_TOKEN)
        word_to_index[SENTENCE_START_TOKEN] = word_count
        index_to_word.append(SENTENCE_END_TOKEN)
        word_to_index[SENTENCE_END_TOKEN] = word_count + 1
        index_to_word.append(UNKNOWN_TOKEN)
        word_to_index[UNKNOWN_TOKEN] = word_count + 2

        # Add word embeddings for special tokens. Values don't matter as they carry no semantic meaning
        word_dim = np.size(embed_layer, 1)
        embed_layer = np.vstack((embed_layer, np.ones(shape=(1, word_dim))))
        embed_layer = np.vstack((embed_layer, 2 * np.ones(shape=(1, word_dim))))
        embed_layer = np.vstack((embed_layer, np.zeros(shape=(1, word_dim))))

    # Replace all words not in our vocabulary with the unknown token
    # Keep track of the unknown token ratio
    unk_count = 0.0
    total = 0.0
    for i, sent in enumerate(tokenized_sentences):
        idx = 0
        for w in sent:
            if w in word_to_index:
                nw = w
            else:
                nw = UNKNOWN_TOKEN
                unk_count += 1.0
            total += 1.0
            tokenized_sentences[i][idx] = nw
            idx += 1
    print("%s unknown tokens/ %s tokens " % (int(unk_count), int(total)))
    print("Unknown token ratio: %s %%" % (unk_count * 100 / total))

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    return X_train, y_train, word_to_index, index_to_word, embed_layer


def load_data_yahoo(filename="data/nfL6.json", vocabulary_size=2000, min_sent_characters=0, embedding=True, sample_size = None):
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading JSON file...")
    exchanges = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for qa in data:
            exchanges.append(("%s %s %s %s" % (SENTENCE_START_TOKEN, qa['question'].lower(), qa['answer'].lower(), SENTENCE_END_TOKEN)))

    # Tokenize the sentences into words
    print("Tokenizing...")
    tokenized_exchanges = [nltk.word_tokenize(sent) for sent in exchanges]

    # Reduce the exchanges to the sample size
    if sample_size is not None:
        tokenized_exchanges = random.sample(tokenized_exchanges, sample_size)

    print("Parsed %d exchanges." % (len(tokenized_exchanges)))

    embed_layer = None
    if embedding:
        print("Using vocabulary size %d." % vocabulary_size)
        embed_layer, word_to_index, index_to_word = load_embedding(vocabulary_size)

        # Add special tokens to vocab
        word_count = len(index_to_word)
        index_to_word.append(SENTENCE_START_TOKEN)
        word_to_index[SENTENCE_START_TOKEN] = word_count
        index_to_word.append(SENTENCE_END_TOKEN)
        word_to_index[SENTENCE_END_TOKEN] = word_count + 1
        index_to_word.append(UNKNOWN_TOKEN)
        word_to_index[UNKNOWN_TOKEN] = word_count + 2

        # Add word embeddings for special tokens. Values are random as they carry no semantic meaning
        word_dim = np.size(embed_layer, 1)
        embed_layer = np.vstack((embed_layer, np.random.rand(1, word_dim)))
        embed_layer = np.vstack((embed_layer, np.random.rand(1, word_dim)))  # TODO Embed meaning for SENTENCE_END_TOKEN
        embed_layer = np.vstack((embed_layer, np.random.rand(1, word_dim)))

    # Replace all words not in our vocabulary with the unknown token
    # Keep track of the unknown token ratio
    unk_count = 0.0
    total = 0.0
    for i, sent in enumerate(tokenized_exchanges):
        idx = 0
        for w in sent:
            if w in word_to_index:
                nw = w
            else:
                nw = UNKNOWN_TOKEN
                unk_count += 1.0
            total += 1.0
            tokenized_exchanges[i][idx] = nw
            idx += 1
    print("%s unknown tokens / %s tokens " % (int(unk_count), int(total)))
    print("Unknown token ratio: %s %%" % (unk_count * 100 / total))

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_exchanges])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_exchanges])

    return X_train, y_train, word_to_index, index_to_word, embed_layer
