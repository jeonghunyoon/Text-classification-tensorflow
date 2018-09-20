"""
Data helper :
    - Data preprocessing
    - Using pandas library
"""

import pandas as pd
import numpy as np
import consts
import gc
import typing

from collections import Counter

from konlpy.tag import Hannanum

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

    print('%s pos document parsing completed.' % _type)

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

    print('%s neg document parsing completed.' % _type)

    nouns_doc_f.close()
    nouns_label_f.close()
    morphs_doc_f.close()
    morphs_label_f.close()


def parse_text_data():
    """Parsing and saving data"""
    train_df = pd.read_csv(consts.TRAIN_DATA_REPO, delimiter='\t')
    test_df = pd.read_csv(consts.TEST_DATA_REPO, delimiter='\t')
    print(train_df.columns)  # id, document, label

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
        document.extend(consts.MAX_SEQUENCE_LENGTH - len(document)) * [consts.PAD_WORD]
    else:
        document = document[:consts.MAX_SEQUENCE_LENGTH]
    return np.array(document)


def get_lookup_dict(documents: np.ndarray):
    corpus = Counter(documents.flatten()).most_common()
    lookup_dict = {}
    rev_lookup_dict = {}
    for idx, val in enumerate(corpus):
        lookup_dict[val[0]] = idx
        rev_lookup_dict[idx] = [val[0]]
    return lookup_dict, rev_lookup_dict


def get_num_seq(padded_seq, lookup_dict):
    """Sequence of string -> Sequence of index"""
    num_seq = []
    for word in padded_seq:
        num_seq.append(lookup_dict[word])
    return num_seq

