from utils import *
from data_utils import *
from argparse import ArgumentParser
from lstm_theano import LSTMTheano


def start(wrapper_path, input_file, modelClass):
    wrapper = load_wrapper(wrapper_path, modelClass)
    with open(input_file, 'rt') as f:
        queries = f.readlines()
    for query in queries:
        print('Q: ' + query, end='')
        response = generate_response(query, wrapper.model,
                                     index_to_word=wrapper.index_to_word, word_to_index=wrapper.word_to_index)
        print_sentence(response, wrapper.index_to_word)
        print()


parser = ArgumentParser()
parser.add_argument('-w', '--wrapper-path', help='The path to the wrapper file', dest='wrapper_path', required=True)
parser.add_argument('-i', '--input-file', help='The file containing question input',
                    default='queries.txt', dest='input_file')
parser.add_argument('-mc', '--model-class', help='The class of the model',
                    default=LSTMTheano, dest='modelClass')

args = parser.parse_args()
start(args.wrapper_path, args.input_file, args.modelClass)
