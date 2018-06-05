### Code for RNN and Spoken Language Understanding

Based on the papers:

[Grégoire Mesnil, Xiaodong He, Li Deng and Yoshua Bengio - **Investigation of Recurrent Neural Network Architectures and Learning Methods for Spoken Language Understanding**](http://www.iro.umontreal.ca/~lisa/pointeurs/RNNSpokenLanguage2013.pdf)

[Grégoire Mesnil, Yann Dauphin, Kaisheng Yao, Yoshua Bengio, Li Deng, Dilek Hakkani-Tur, Xiaodong He, Larry Heck, Gokhan Tur, Dong Yu and Geoffrey Zweig - **Using Recurrent Neural Networks for Slot Filling in Spoken Language Understanding**](http://www.iro.umontreal.ca/~lisa/pointeurs/taslp_RNNSLU_final_doubleColumn.pdf)

## Code

This code allows to get state-of-the-art results and a significant improvement
(+1% in F1-score) with respect to the results presented in the paper.

In order to reproduce the results, make sure Theano is installed and the
repository is in your `PYTHONPATH`, e.g run the command

`export PYTHONPATH=/path/where/rnn_slu/is:$PYTHONPATH`.

File Formats:
-------------

The files must have a word token and corresponding label at each line. Between each sentences there must be a blank line

e.g.:

what   O
aircraft   O
is   O
used   O
on   O
delta   B-airline_name
flight   O
DIGITDIGITDIGITDIGIT   B-flight_number
from   O
kansas   B-fromloc.city_name
city   I-fromloc.city_name
to   O
salt   B-toloc.city_name
lake   I-toloc.city_name
city   I-toloc.city_name

i   O
want   O
to   O
go   O
from   O
boston   B-fromloc.city_name
to   O
atlanta   B-toloc.city_name
on   O
monday   B-depart_date.day_name

Output File:
------------

The output file contains each word token at each line with its gold label and predicted label. Contains a blank line between sentences. This file contains also BOS (beginning of sentence) and EOS (end of sentence) tokens so may not match the length of the input file.

e.g.:
BOS O O
monday B-depart_date.day_name O
morning B-depart_time.period_of_day O
i O O
would O O
like O O
to O O
fly O O
from O O
columbus B-fromloc.city_name B-fromloc.city_name
to O O
indianapolis B-toloc.city_name B-toloc.city_name
EOS O O

BOS O O
on O O
wednesday B-depart_date.day_name O
april B-depart_date.month_name O
sixth B-depart_date.day_number O
i O O
would O O
like O O
to O O
fly O O
from O O
long B-fromloc.city_name O
beach I-fromloc.city_name O
to O O
columbus B-toloc.city_name O
after B-depart_time.time_relative O
DIGIT B-depart_time.time I-toloc.city_name
pm I-depart_time.time O
EOS O O



Configuration file:
-------------------

The following parameters can be set by using the configuration file: (See the config.cfg file)

lr: 0.1
win: 7
bs: 10
nhidden: 100
seed: 345
emb_dimension: 100
nepochs: 5

lr: starting learning rate
win: context window size
bs: mini batch size
nhidden: size of the hidden layer
seed: random seed
emb_dimension: dimension of the word embeddings
nepochs: maximum number of backpropagation steps


Training Elman RNN:
-------------------

python rnn_slu/lus/rnn_elman_train.py <training_data> <validation_data> <word_dictionary> <label_dictionary> <config_file> <model_directory>

e.g.:
python rnn_slu/lus/rnn_elman_train.py rnn_slu/data/train_atis_samp.txt rnn_slu/data/dev_atis.txt rnn_slu/data/word_dict.txt rnn_slu/data/label_dict.txt rnn_slu/config.cfg model_elman

Testing Elman RNN:
------------------

python rnn_slu/lus/rnn_elman_test.py <model_directory> <test_file> <word_dictionary> <label_dictionary> <config_file> <output_file>

e.g.:
python rnn_slu/lus/rnn_elman_test.py model_elman rnn_slu/data/test_atis.txt rnn_slu/data/word_dict.txt rnn_slu/data/label_dict.txt rnn_slu/config.cfg test_out.txt

Training Jordan RNN:
--------------------

python rnn_slu/lus/rnn_jordan_train.py <training_data> <validation_data> <word_dictionary> <label_dictionary> <config_file> <model_directory>

e.g.:
python rnn_slu/lus/rnn_jordan_train.py rnn_slu/data/train_atis_samp.txt rnn_slu/data/dev_atis.txt rnn_slu/data/word_dict.txt rnn_slu/data/label_dict.txt rnn_slu/config.cfg model_elman

Testing Jordan RNN:
-------------------

python rnn_slu/lus/rnn_jordan_test.py <model_directory> <test_file> <word_dictionary> <label_dictionary> <config_file> <output_file>

e.g.:
python rnn_slu/lus/rnn_jordan_test.py model_elman rnn_slu/data/test_atis.txt rnn_slu/data/word_dict.txt rnn_slu/data/label_dict.txt rnn_slu/config.cfg test_out.txt
