from DateTime.DateTime import time
from gensim.matutils import Sparse2Corpus
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm


def count_vectorizer(max_features=400):
    t0 = time()
    tf_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    # lowercase=True,
                                    stop_words='english',
                                    max_features=max_features
                                    # vocabulary=[(k+1,v) for k,v in dictionary]
                                    )

    print("done in %0.6fs." % (time() - t0))

    return tf_vectorizer


# def tfidf_vectorizer():
#     t0 = time()
#     tf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),
#                                     # lowercase=True,
#                                     stop_words='english'
#                                     # vocabulary=dictionary
#
#                                     )
#
#     print("done in %0.3fs." % (time() - t0))
#
#     return tf_vectorizer


def transform_text(vectorizer, text):
    return vectorizer.transform(text)


def fit_to_vectorizer(vectorizer, text):
    return vectorizer.fit_transform(text)


# def get_bow_representation(dictionary, text):
#     # convert dictionary into bag-of-words representation
#     word_bag = [dictionary.doc2bow(row) for row in text]
#     # print ">> type word bag"
#     # print type(word_bag)
#     return word_bag


def map_idvec2word(vectorizer):
    return {v: k for k, v in tqdm(vectorizer.vocabulary_.items())}


def convert_to_sparse_bow(bow_vectorizer):
    return Sparse2Corpus(bow_vectorizer, documents_columns=False)


def get_feature_names(vectorizer):
    return vectorizer.get_feature_names()

