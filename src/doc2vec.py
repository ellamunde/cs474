# coding=utf-8
from gensim.models import word2vec, Doc2Vec
# from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from tqdm import tqdm

import logging
import random
import os
import sklearn.metrics.pairwise as pairwise

# progress bar
from preprocessing import get_root_dir

tqdm.pandas(desc="progress-bar")
# format logging configuration
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def build_doc2vec_model(min_count=1, window=2, size=100, sample=1e-4, negative=5, workers=8):
    """
    :param min_count: ignore all words with total frequency lower than this. You have to set this to 1, since the sentence labels only appear once. Setting it any higher than 1 will miss out on the sentences.
    :param window: the maximum distance between the current and predicted word within a sentence. Word2Vec uses a skip-gram model, and this is simply the window size of the skip-gram model.
    :param size: dimensionality of the feature vectors in output. 100 is a good number. If you’re extreme, you can go up to around 400.
    :param sample: threshold for configuring which higher-frequency words are randomly downsampled
    :param negative:
    :param workers: use this many worker threads to train the model
    :return:
    """
    return Doc2Vec(min_count=min_count,
                   window=window,
                   size=400,
                   sample=1e-4,
                   negative=5,
                   workers=8
                   )


def get_sentences(text, lable):
    sentences = []
    for i in range(len(lable)):
        sentences += [text[i], lable[i]]
    return sentences


def sentences_perm(sentences):
    shuffled = list(sentences)
    random.shuffle(shuffled)
    return shuffled


def train_model(model, sentences, loops=10):
    for epoch in range(loops):
        model.train(tqdm(sentences_perm(sentences)))

    save_model(model)
    return model


def infer_vector(model, sentence):
    return model.infer_vector(sentence)


def save_model(model):
    model.save(os.path.abspath(get_root_dir() + "/model/d2v_model"))


def cosine_similarity(sentence1, sentence2):
    return pairwise.cosine_similarity(sentence1, sentence2)


def prediction(model, test):
    for idx, row in test.itterows():
        model.most_similar()


def process_test_data(f):
    """
    for splitting the sentence
    :param f:
    :return:
    """
    yield [simple_preprocess(line) for i, line in enumerate(f)]


def get_word_distribution(model, text, topic=False):
    # for doc_id in range(len(sents_arr)):
    if not topic:
        inferred_vector = model.infer_vector(text.words)
    else:
        inferred_vector = model.docvecs[text]
    print ">> inferred vector"
    print inferred_vector
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    print ">> sims"
    print sims
    return sims


def build_matrix_csr_train(model, sentences, topics):
    topic_list = {}

    vocabulary = model.wv.index2word
    idx_range = len(model.wv.index2word)
    len_text = len(topics)
    csrmatrix = None

    for i in tqdm(range(len_text)):
        text_nth = [0.0] * idx_range
        bow_topic = get_word_distribution(model, sentences[i])[0]
        for i in bow_topic:
            print bow_topic

        # for idx in tqdm(range(len(column))):
        #     name = get_feature_names(vectorizer)[idx]
        #     # print ">> name"
        #     # print column[idx]
        #     # print value[idx]
        #     # print name
        #
        #     value[idx] = topic_words_dist[topicno][name] * value[idx] if name in topic_words_dist[topicno].keys() else 0
        #     # print len(text_nth), " < ", column[idx]
        #
        #     if len(text_nth) > column[idx]:
        #         text_nth[column[idx]] = value[idx]
        #
        # temp = csr_matrix(text_nth)
        # if csrmatrix is not None:
        #     csrmatrix = vstack([csrmatrix, temp])
        # else:
        #     csrmatrix = temp
        #
        # # x_nth.extend([i] * idx_range)
        # # y_nth.extend(k for k in range(idx_range))
        # # -- FOR DEVELOPMENT ONLY --
        # # break

    # print csrmatrix
    return csrmatrix


def build_matrix_csr(model, sentences, topics):
    topic_list = {}

    vocabulary = model.wv.index2word
    idx_range = len(model.wv.index2word)
    len_text = len(topics)
    csrmatrix = None

    for i in tqdm(range(len_text)):
        text_nth = [0.0] * idx_range

        # print ">> topic"
        # print topics[i]
        if topics[i] in topic_list.keys():
            topicno = topic_list[topics[i]]
        else:
            bow_topic = get_word_distribution(model, topics[i])[0]
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


class LabeledLineSentence(object):
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __iter__(self):
        # yield [TaggedDocument(simple_preprocess(doc), [self.label[idx]]) for idx, doc in enumerate(self.text)]
        # yield [TaggedDocument(doc, [self.label[idx]]) for idx, doc in enumerate(self.text)]
        for idx, doc in enumerate(self.text):
            # print ">> sentences"
            # print doc
            # print self.label[idx]
            yield TaggedDocument(simple_preprocess(doc), [self.label[idx]])
