"""
==========================================================================
command : python executor.py {is_first_time} {parse_type} {embedding_type}
==========================================================================
"""
import sys
import consts
import logging
import cnn
from data_helpers import get_input
from word_embedding import WordEmbedding
from hyper_params import HyperParams
from experiment import ClassifierExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Command line arguments
args = sys.argv
is_first_time = args[1]
parse_type = args[2]
embedding_type = args[3]

# 2. Loading the data
train_seqs, train_y, test_seqs, test_y = get_input(is_first_time=is_first_time,
                                                   parse_type=parse_type)

hparams = HyperParams().get_cnn_hyper_params()

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
train_y = train_y.reshape([-1, consts.NUM_LABELS])
test_X = test_X.reshape([-1, consts.MAX_SEQUENCE_LENGTH, hparams.embedding_dim])
test_y = test_y.reshape([-1, consts.NUM_LABELS])

# train_X: (149995, 100, 256)
# train_Y: (149995, 2)
# test_X: (49997, 100, 256)
# test_Y: (49997, 2)
logger.info({'train_X.shape': train_X.shape, 'train_Y.shape': train_y.shape,
             'test_X.shape': test_X.shape, 'test_Y.shape': test_y.shape})

# 5. Train
cnn_experiment = ClassifierExperiment(train_X, test_X, train_y, test_y, hparams, cnn.model_fn)
cnn_experiment.run_train_and_evaluate(False)
