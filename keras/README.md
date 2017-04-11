An implementation of sequence-to-sequence question answering using LSTM with Keras. 

To run Keras using Theano backend, copy the .keras folder to your home folder.

Get the Glove word vector embeddings [here](http://nlp.stanford.edu/data/glove.6B.zip) (800MB). Extract glove.6B.100d.txt to /data or specify a different file by changing the value of EMBEDDING_PATH in /data_utils.py.

### Model structure
![Model Image](docs/EncDecModel.png)