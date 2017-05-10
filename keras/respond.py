from argparse import ArgumentParser

from lstm import utils
from lstm.enc_dec import LSTMEncDec


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='md', required=True,
                        help='The path to the model directory')
    parser.add_argument('-q', '--queries', dest='queries', required=False, default=None,
                        help='The text file containing queries. If not set, runs in shell mode')
    parser.add_argument('-l', '--log', dest='log', required=False, default='conversation.txt',
                        help='The file to which the conversation is logged. Defaults to log.txt. Logging only works if the ' +
                             'application exits properly.')
    arg = parser.parse_args()
    return arg

args = parse_args()

print('Loading model...')
model = utils.load_model(args.md, LSTMEncDec)

conv = []

if args.queries is not None:
    with open(args.queries, 'rt') as f:
        queries = f.readlines()
        f.close()
    print('Showing responses for %s' % args.queries)
    for q in queries:
        r = model.generate_response(q.lower()) + '\n'
        print(q + r)
        conv.append(q + r)
else:
    print('Running in shell mode. Type \'exit\' to terminate...')
    while True:
        q = input('Q: ')
        if q == 'exit':
            break
        r = model.generate_response(q.lower()) + '\n'
        print(r)
        conv.append(q + '\n' + r)

with open(args.log, 'wt') as f:
    f.write('\n\n'.join(conv))
    f.close()
