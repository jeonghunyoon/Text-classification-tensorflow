# text-classification-cnn-tf (Sentiment analysis)

This project is **Sentiment Analysis** for Naver(Korean portal site) movie review.

It uses **Tensorflow highlevel api** and `CNN`, `RNN` models for classification.

For simple keras implementation, please refer to 
https://github.com/jeonghunyoon/spam-classification-cnn-tf.



## Data
https://github.com/e9t/nsmc


## References
This project is based on following papers.
- http://www.aclweb.org/anthology/D14-1181
- https://arxiv.org/abs/1510.03820


## Run project
```python
python executor.py is_first_time parse_type embedding_type
# command example : python executor.py false morphs doc2vec
```
- is_first_time : <U>*true*</U> or <U>*false*</U>. When <U>*true*</U> it will create all parsed files(morphs, nouns) 
and store it. When <U>*false*</U> it just load the parsed data according to parse_type.

- parse_type : <U>*morphs*</U> or <U>*nouns*</U>

- embedding_type : <U>*doc2vec*</U> or <U>*word2vec*</U>. LDA, NMF will be added.
