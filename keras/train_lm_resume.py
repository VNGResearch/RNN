from argparse import ArgumentParser

from lstm.lang_model import *
from settings_lm import *
from utils import *

__LOGGER = logging.getLogger()
__LOGGER.level = logging.INFO


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='md', required=True,
                        help='The path to the model directory')
    parser.add_argument('-q', '--queries', dest='queries', required=False, default='queries.txt',
                        help='The text file containing queries. If not set, runs in shell mode')
    parser.add_argument('-d', '--dataset', dest='ds', required=False, default='opensub',
                        help='''The training dataset to be used. 
                        Defaults to the OpenSubtitle dataset. 
                        Values: opensub, shakespeare, yahoo, southpark, cornell, vnnews''')
    arg = parser.parse_args()
    return arg


args = parse_args()

loader, switch = get_loader(args.ds)
if switch:
    DOC_COUNT = DATA_SIZE

logging.info('Creating model...')
model = LSTMLangModel.load(args.md)

word_vec, word_to_index, index_to_word = load_embedding(VOCABULARY_SIZE,
                                                        embed_path=EMBEDDING_PATH,
                                                        embed_type=EMBEDDING_TYPE)
X, y, samples, output_mask = loader(word_vec, word_to_index, index_to_word,
                                    sample_size=DOC_COUNT,
                                    sequence_len=SEQUENCE_LENGTH,
                                    vec_labels=False)
y, output_mask = generate_lm_labels(X, word_to_index)

X_val = X[:VAL_SPLIT]
X_train = X[VAL_SPLIT:]
y_val = y[:VAL_SPLIT]
y_train = y[VAL_SPLIT:]
del X, y

model.train(X_train, y_train, N_EPOCH, batch_size=BATCH_SIZE, Xval=X_val, yval=y_val,
            train_mask=output_mask[VAL_SPLIT:], val_mask=output_mask[:VAL_SPLIT])
