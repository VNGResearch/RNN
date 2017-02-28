from lstm_keras import *
from argparse import ArgumentParser
import utils


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='md', required=True,
                        help='The path to the model directory')
    parser.add_argument('-q', '--queries', dest='queries', required=False, default=None,
                        help='The text file containing queries. If not set, runs in shell mode')
    parser.add_argument('-c', '--class', dest='cls_idx', required=False, default='1',
                        help='The model class. Values: 1 (Word-vector based model), 2 (Probability model). Defaults to 1.')
    arg = parser.parse_args()
    return arg

args = parse_args()

print('Loading model...')
cls = LSTMEncDec if args.cls_idx == '1' else LSTMEncDec2
model = utils.load_model(args.md, cls)

if args.queries is not None:
    with open(args.queries, 'rt') as f:
        queries = f.readlines()
        f.close()
    print('Showing responses from %s' % args.queries)
    for q in queries:
        r = model.generate_response(q)
        print(q + r + '\n')
else:
    print('Running in shell mode. Type \'exit\' to terminate...')
    while True:
        q = input('Q: ')
        if q == 'exit':
            break
        r = model.generate_response(q)
        print(r + '\n')
