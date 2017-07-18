from argparse import ArgumentParser

from lstm.lang_model import LSTMLangModel


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='md', required=True,
                        help='The path to the model directory')
    parser.add_argument('-q', '--queries', dest='queries', required=False, default=None,
                        help='The text file containing queries. If not set, runs in shell mode.')
    parser.add_argument('-k', '--topk', dest='k', required=False, default=5,
                        help='The number of word candidate to output.')
    parser.add_argument('-l', '--log', dest='log', required=False, default='lm_log.txt',
                        help='''The file to which the conversation is logged. Defaults to lm_log.txt. 
                        Logging only works if the application exits properly.''')
    arg = parser.parse_args()
    return arg


def pred_to_str(prediction):
    s = ''
    for p in prediction:
        s += '%s: %.6f\n' % (p[0], p[1])

    return s


args = parse_args()

print('Loading model...')
model = LSTMLangModel.load(args.md)
conv = []

if args.queries is not None:
    with open(args.queries, 'rt') as f:
        queries = f.readlines()
        f.close()
    print('Showing responses for %s' % args.queries)
    for q in queries:
        r = pred_to_str(model.predict(q.lower(), top=args.k)) + '\n'
        print(q + r)
        conv.append(q + r)
else:
    print('Running in shell mode. Type \'exit\' to terminate...')
    while True:
        q = input('Q: ')
        if q == 'exit':
            break
        r = pred_to_str(model.predict(q.lower(), top=args.k)) + '\n'
        print(r)
        conv.append(q + '\n' + r)

with open(args.log, 'wt') as f:
    f.write('\n'.join(conv))
    f.close()
