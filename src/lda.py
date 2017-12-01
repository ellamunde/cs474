from scipy.sparse import coo_matrix, dok_matrix, hstack, vstack, csr_matrix
from tqdm import tqdm
from time import time
from gensim import corpora, matutils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

import gensim
import os
import preprocessing
import wordnet
import numpy
from pprint import pprint


def preprocess(text):
    join = " ".join
    tokens = [preprocessing.extract_tokens(row) for row in tqdm(text)]
    lemma = [wordnet.lemmatize_words(x) for x in tqdm(tokens)]

    no_stopwords = []
    no_stopwords_sent = []
    for x in tqdm(lemma):
        # print x
        x = preprocessing.remove_stopwords(
            preprocessing.get_tokens_only(x)
        )
        no_stopwords.append(x)
        no_stopwords_sent.append(join(x))
        # print x

    return no_stopwords, no_stopwords_sent


def map_idvec2word(vectorizer):
    return {v: k for k, v in tqdm(vectorizer.vocabulary_.items())}


def process_to_bow(vectorizer, lda_model, text):
    # print text
    # print type(text)
    corpus = convert_to_lda_bow(transform_text(vectorizer, [text]))
    bow_list = [tup for tup in tqdm(list(lda_model[corpus]))]

    return bow_list


def get_words_topic(lda_model, total_topic, dict_len):
    topic_words = []
    for i in tqdm(range(total_topic)):
        topic_words.append(dict(lda_model.show_topic(topicid=i, topn=dict_len)))

    return topic_words


def process_to_get_topicno(lda_model, bow_topic, get='doc'):
    # print bow_topic
    doc_topics, word_topics, phi_values = lda_model.get_document_topics(bow_topic, per_word_topics=True)
    if get == 'phi':
        return sorted(doc_topics, key=lambda tup: tup[1], reverse=True)[0][0]
    elif get == 'word':
        return sorted(word_topics, key=lambda tup: tup[1], reverse=True)[0][0]
    else:
        return sorted(phi_values, key=lambda tup: tup[1], reverse=True)[0][0]


def convert_to_lda_bow(bow_vectorizer):
    return matutils.Sparse2Corpus(bow_vectorizer, documents_columns=False)


def get_feature_names(vectorizer):
    return vectorizer.get_feature_names()


def get_dictionary(text):
    # print text
    dictionary = corpora.Dictionary(text)
    # print(dictionary.token2id)
    return dictionary


def load_dictionary(directory):
    return corpora.Dictionary.load(directory)


def get_bow_representation(dictionary, text):
    # convert dictionary into bag-of-words representation
    word_bag = [dictionary.doc2bow(row) for row in text]
    # print ">> type word bag"
    # print type(word_bag)
    return word_bag


def load_lda_model(directory):
    if os.path.exists(directory):
        return gensim.models.LdaModel.load(directory)


def count_vectorizer():
    t0 = time()
    tf_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    # lowercase=True,
                                    stop_words='english'
                                    # vocabulary=[(k+1,v) for k,v in dictionary]
                                    )

    print("done in %0.6fs." % (time() - t0))

    return tf_vectorizer


def tfidf_vectorizer():
    t0 = time()
    tf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                    # lowercase=True,
                                    stop_words='english'
                                    # vocabulary=dictionary

                                    )

    print("done in %0.3fs." % (time() - t0))

    return tf_vectorizer


def transform_text(vectorizer, text):
    return vectorizer.transform(text)


def fit_to_vectorizer(vectorizer, text):
    return vectorizer.fit_transform(text)


def build_lda_model(word_bag, dictionary, num_topics, alpha, passes):
    lda_model = gensim.models.ldamodel.LdaModel(word_bag,
                                                num_topics=num_topics,
                                                id2word=dictionary,
                                                passes=passes,
                                                alpha=alpha
                                                )
    return lda_model


def save_lda_model(model, directory):
    model.save(directory)


def build_matrix_csr(vectorizer, lda_model, topic_words_dist, topics, texts):
    topic_list = {}

    idx_range = len(vectorizer.vocabulary_)
    # print ">> vocabulary range"
    # print idx_range
    len_text = len(topics)

    x_nth = []
    y_nth = []
    csrmatrix = None

    for i in tqdm(range(len_text)):
        text_nth = [0.0] * idx_range

        # print ">> topic"
        # print topics[i]
        if topics[i] in topic_list.keys():
            topicno = topic_list[topics[i]]
        else:
            bow_topic = process_to_bow(vectorizer, lda_model, topics[i])[0]
            topicno = sorted(bow_topic, key=lambda tup: tup[1], reverse=True)[0][0]
            # print ">> bow topic"
            # print bow_topic
            topic_list[topics[i]] = topicno
        # print ">> topic no"
        # print topicno

        txt = transform_text(vectorizer, [texts[i]])
        # print type(txt)
        # print ">> matrix"
        # print txt
        txt = txt.tolil(txt)
        value = txt.data[0]
        column = txt.rows[0]
        # print ">> value"
        # print value
        # print ">> column"
        # print column

        for idx in tqdm(range(len(column))):
            name = get_feature_names(vectorizer)[idx]
            # print ">> name"
            # print column[idx]
            # print value[idx]
            # print name

            value[idx] = topic_words_dist[topicno][name] * value[idx] if name in topic_words_dist[topicno].keys() else 0
            # print len(text_nth), " < ", column[idx]

            if len(text_nth) > column[idx]:
                text_nth[column[idx]] = value[idx]

        temp = csr_matrix(text_nth)
        if csrmatrix is not None:
            csrmatrix = vstack([csrmatrix, temp])
        else:
            csrmatrix = temp

        # x_nth.extend([i] * idx_range)
        # y_nth.extend(k for k in range(idx_range))
        # -- FOR DEVELOPMENT ONLY --
        # break

    # print csrmatrix
    return csrmatrix

