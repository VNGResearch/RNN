import gc
from argparse import ArgumentParser

from lstm.data_utils import *

from lstm.enc_dec import *
from settings_enc_dec import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='md', required=True,
                        help='The path to the model directory')
    parser.add_argument('-q', '--queries', dest='queries', required=False, default='queries.txt',
                        help='The text file containing queries. If not set, runs in shell mode')
    parser.add_argument('-d', '--dataset', dest='ds', required=False, default='opensub',
                        help='''The training dataset to be used. 
                        Defaults to the OpenSubtitle dataset. 
                        Values: opensub, shakespeare, yahoo, southpark, cornell''')
    arg = parser.parse_args()
    return arg


args = parse_args()

loader, switch = get_loader(args.ds)
if switch:
    DOC_COUNT = DATA_SIZE

print('Loading model...')
model = utils.load_model(args.md, LSTMEncDec)

X, y, word_to_index, index_to_word, word_vec, samples, output_mask = loader(vocabulary_size=len(model.index_to_word),
                                                                            sample_size=DOC_COUNT,
                                                                            sequence_len=model.sequence_len,
                                                                            vec_labels=(model.out_type == 0))
# Get queries
with open(QUERY_FILE, 'rt') as f:
    queries = [q.rstrip() for q in f.readlines()]
    f.close()
queries.extend(samples)

X_val = X[:VAL_SPLIt]
X_train = X[VAL_SPLIt:]
y_val = y[:VAL_SPLIt]
y_train = y[VAL_SPLIt:]
X = y = None
gc.collect()

model.train(X_train, y_train, N_EPOCH, batch_size=BATCH_SIZE, queries=queries, Xval=X_val, yval=y_val,
            train_mask=output_mask[VAL_SPLIt:], val_mask=output_mask[:VAL_SPLIt])
