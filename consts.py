"""
Constants
"""

TRAIN_DATA_REPO = './input/ratings_train.txt'
TEST_DATA_REPO = './input/ratings_test.txt'

# Original data
MORPHS_TRAIN_DOC_REPO = './input/morphs_ratings_train.tsv'
NOUNS_TRAIN_DOC_REPO = './input/nouns_ratings_train.tsv'
MORPHS_TEST_DOC_REPO = './input/morphs_ratings_test.tsv'
NOUNS_TEST_DOC_REPO = './input/nouns_ratings_test.tsv'

# Because of performance issue, parse and save.
# MORPHS_TRAIN_DOCUMENTS = './input/morphs_train_documents.txt'
# MORPHS_TRAIN_LABELS = './input/morphs_train_labels.txt'
# MORPHS_TEST_DOCUMENTS = './input/morphs_test_documents.txt'
# MORPHS_TEST_LABELS = './input/morphs_test_labels.txt'
# NOUNS_TRAIN_DOCUMENTS = './input/nouns_train_documents.txt'
# NOUNS_TRAIN_LABELS = './input/nouns_train_labels.txt'
# NOUNS_TEST_DOCUMENTS = './input/nouns_test_documents.txt'
# NOUNS_TEST_LABELS = './input/nouns_test_labels.txt'

TRAIN_DOCUMENTS = './input/{}_train_documents.txt'
TRAIN_LABELS = './input/{}_train_labels.txt'
TEST_DOCUMENTS = './input/{}_test_documents.txt'
TEST_LABELS = './input/{}_test_labels.txt'

PAD_WORD = '<PAD>'
MAX_SEQUENCE_LENGTH = 100
NUM_LABELS = 2
