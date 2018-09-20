"""
Data helper :
    - Data preprocessing
    - Using pandas library
"""

import pandas as pd
import numpy as np
import consts
import logging
from konlpy.tag import Hannanum
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Just for debugging.


class ReshapedLabelEncoder(LabelEncoder):
    """For using the Pipeline class, we reshape the result."""

    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)


def parse(df, _type: str):
    """Parse function"""

    # Parser
    korean_parser = Hannanum()

    neg = df[df['label'] == 0]['document'].tolist()
    pos = df[df['label'] == 1]['document'].tolist()

    nouns_doc_f = open('./input/nouns_{}_documents.txt'.format(_type), 'w')
    nouns_label_f = open('./input/nouns_{}_labels.txt'.format(_type), 'w')
    morphs_doc_f = open('./input/morphs_{}_documents.txt'.format(_type), 'w')
    morphs_label_f = open('./input/morphs_{}_labels.txt'.format(_type), 'w')

    logger.info("Starting parsing...")
    for doc in neg:
        try:
            nouns_doc_f.write(','.join(korean_parser.nouns(doc)) + '\n')
            nouns_label_f.write('{}\n'.format(0))
        except:
            pass
        try:
            morphs_doc_f.write(','.join(korean_parser.morphs(doc)) + '\n')
            morphs_label_f.write('{}\n'.format(0))
        except:
            pass

    logger.info('%s pos document parsing completed.' % _type)

    for doc in pos:
        try:
            nouns_doc_f.write(','.join(korean_parser.nouns(doc)) + '\n')
            nouns_label_f.write('{}\n'.format(1))
        except:
            pass
        try:
            morphs_doc_f.write(','.join(korean_parser.morphs(doc)) + '\n')
            morphs_label_f.write('{}\n'.format(1))
        except:
            pass

    logger.info('%s neg document parsing completed.' % _type)

    nouns_doc_f.close()
    nouns_label_f.close()
    morphs_doc_f.close()
    morphs_label_f.close()


def parse_text_data():
    """Parsing and saving data"""
    train_df = pd.read_csv(consts.TRAIN_DATA_REPO, delimiter='\t')
    test_df = pd.read_csv(consts.TEST_DATA_REPO, delimiter='\t')

    parse(train_df, 'train')
    parse(test_df, 'test')


def get_docs_n_labels_from_file(doc_path, label_path):
    """Return documents and labels from the path."""
    docs = []
    labels = []
    with open(doc_path, 'r') as f:
        for doc in f:
            docs.append(doc.strip().split(','))
    with open(label_path, 'r') as f:
        for label in f:
            labels.append(int(label.strip()))
    return docs, labels


def get_padded_seq(document):
    """Doc(sequence of string) -> Sequence with '<PAD>' and max length."""
    if len(document) < consts.MAX_SEQUENCE_LENGTH:
        document.extend((consts.MAX_SEQUENCE_LENGTH - len(document)) * [consts.PAD_WORD])
    else:
        document = document[:consts.MAX_SEQUENCE_LENGTH]
    return np.array(document)


def get_one_hot_labels(labels):
    """Using pipeline, transform label data into one hot vector."""
    reshaped_label_encoder = ReshapedLabelEncoder()
    one_hot_encoder = OneHotEncoder()
    pipeline = Pipeline([
        ('label_encoder', reshaped_label_encoder),
        ('one_hot_encoder', one_hot_encoder)
    ])
    one_hot_vector = pipeline.fit_transform(labels)

    return one_hot_vector.toarray()


def get_input(is_first_time, parse_type):
    """
    In step 1, store data after parsing for performance issues.
    This process is needed at the first time only.

    In step 2, use data according to parse_type(morphs, nouns)
    """

    # 1
    if is_first_time == 'true':
        logger.info("First execution...")
        parse_text_data()

    # 2
    train_docs, train_labels = get_docs_n_labels_from_file(
        consts.TRAIN_DOCUMENTS.format(parse_type), consts.TRAIN_LABELS.format(parse_type))
    test_docs, test_labels = get_docs_n_labels_from_file(
        consts.TEST_DOCUMENTS.format(parse_type), consts.TEST_LABELS.format(parse_type))

    # 3
    train_seqs = np.array([get_padded_seq(doc) for doc in train_docs])
    test_seqs = np.array([get_padded_seq(doc) for doc in test_docs])

    # 4
    train_oh_labels = get_one_hot_labels(train_labels)
    test_oh_labels = get_one_hot_labels(test_labels)

    train_seqs, train_oh_labels = shuffle(train_seqs, train_oh_labels, random_state=43)
    test_seqs, test_oh_labels = shuffle(test_seqs, test_oh_labels, random_state=43)

    return train_seqs, train_oh_labels, test_seqs, test_oh_labels

