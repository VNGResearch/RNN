import utils
import gc

from data_utils import *
from lstm_enc_dec import *
from settings_alt import *
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='md', required=True,
                        help='The path to the model directory')
    parser.add_argument('-q', '--queries', dest='queries', required=False, default=None,
                        help='The text file containing queries. If not set, runs in shell mode')
    parser.add_argument('-c', '--class', dest='cls_idx', required=False, default='1',
                        help='The model class. Values: 1 (Word-vector based model), 2 (Probability model). Defaults to 1.')
    parser.add_argument('-d', '--dataset', dest='ds', required=True, default='opensub',
                        help='The training dataset to be used. Defaults to the OpenSubtitle dataset. Values: opensub, shakespeare, yahoo')
    arg = parser.parse_args()
    return arg


args = parse_args()

if args.ds == 'opensub':
    loader = load_data_opensub
elif args.ds == 'shakespeare':
    loader = load_data_shakespeare
elif args.ds == 'yahoo':
    loader = load_data_yahoo
    DOC_COUNT = DATA_SIZE
else:
    raise ValueError('Invalid dataset.')

print('Loading model...')
cls = LSTMEncDec if args.cls_idx == '1' else LSTMEncDec2
model = utils.load_model(args.md, cls)

X, y, word_to_index, index_to_word, word_vec, samples, output_mask = loader(vocabulary_size=len(model.index_to_word),
                                                                            sample_size=DOC_COUNT,
                                                                            sequence_len=model.sequence_len,
                                                                            vec_labels=False)
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
