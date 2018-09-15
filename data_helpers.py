"""
Data helper :
    - Data preprocessing
    - Using pandas library
"""

import pandas as pd
import consts
import gc

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
    parse(test_df, 'train')
