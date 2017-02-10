from argparse import ArgumentParser
import json
import numpy as np

parser = ArgumentParser()
parser.add_argument('-l', '--length', help='The number of generated queries. Defaults to 100',
                    dest='len', default=100, type=int)
parser.add_argument('-o', help='The output file. Defaults to queries.txt',
                    dest='outfile', default='queries.txt')
args = parser.parse_args()

infile = "data/nfL6.json"
with open(infile, 'r') as fi:
    data = json.load(fi)
    fi.close()

questions = []
for qa in data:
    questions.append(qa['question'])

questions = np.random.permutation(questions)[:args.len]
print('Questions generated:\n')
print('\n'.join(questions))

with open(args.outfile, 'wt') as fo:
    fo.write('\n'.join(questions))
    fo.close()
