"""
==========================================================================
command : python executor.py {is_first_time} {parse_type} {embedding_type}
==========================================================================
"""
import sys
import consts
import logging
from data_helpers import get_input
from word_embedding import WordEmbedding
from hyper_params import HyperParams

logger = logging.getLogger(__name__)

# 1. Command line arguments
args = str(sys.argv)
is_first_time = args[0]
parse_type = args[1]
embedding_type = args[2]

# 2. Loading the data
train_seqs, train_Y, test_seqs, test_Y = get_input(is_first_time=is_first_time,
                                                   parse_type=parse_type)

hparams = HyperParams().get_hyper_params()

# 3. Transform the data using embedding vectors.
embedding = WordEmbedding(train_seqs, hparams.embedding_dim)
if embedding_type == 'doc2vec':
    model = embedding.get_d2v_model()
else:
    model = embedding.get_w2v_model()
train_X = embedding.get_embedding_mtx(model, train_seqs)
test_X = embedding.get_embedding_mtx(model, test_seqs)

# 4. Reshape
train_X = train_X.reshape([-1, consts.MAX_SEQUENCE_LENGTH, hparams.embedding_dim])
train_Y = train_Y.reshape([-1, consts.NUM_LABELS])
test_X = test_X.reshape([-1, consts.MAX_SEQUENCE_LENGTH, hparams.embedding_dim])
test_Y = test_Y.reshape([-1, consts.NUM_LABELS])

logger.info({'train_X.shape': train_X.shape, 'train_Y.shape': train_Y.shape, 'test_X.shape': test_X.shape,
             'test_Y.shape': test_Y.shape})
