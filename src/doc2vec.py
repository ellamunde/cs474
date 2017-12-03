# coding=utf-8
import multiprocessing

from gensim.models import word2vec, Doc2Vec
# from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from pandas import DataFrame
from scipy.sparse import csr_matrix, vstack
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


def build_doc2vec_model_dm(size=400, window=8, min_count=1, sample=1e-4, negative=5, workers=None,  dm=1, dm_concat=1, batch_words=10000):
    """
    :param min_count: ignore all words with total frequency lower than this. You have to set this to 1, since the sentence labels only appear once. Setting it any higher than 1 will miss out on the sentences.
    :param window: the maximum distance between the current and predicted word within a sentence. Word2Vec uses a skip-gram model, and this is simply the window size of the skip-gram model.
    :param size: dimensionality of the feature vectors in output. 100 is a good number. If you’re extreme, you can go up to around 400.
    :param sample: threshold for configuring which higher-frequency words are randomly downsampled
    :param negative:
    :param workers: use this many worker threads to train the model
    :return:
    """
    print ">> -----------------------------"
    print "Doc2Vec DM model specification:"
    print ">> min_count: " + str(min_count)
    print ">> window: " + str(window)
    print ">> size: " + str(size)
    print ">> negative: " + str(negative)
    print ">> workers: " + str(workers)
    print ">> dm: " + str(dm)
    print ">> dm_concat: " + str(dm_concat)
    print ">> batch_words: " + str(batch_words)
    print ">> -----------------------------"

    if workers is None:
        workers = multiprocessing.cpu_count()

    return Doc2Vec(size=size,
                   window=window,
                   min_count=min_count,
                   sample=sample,
                   negative=negative,
                   workers=workers,
                   dm=dm,
                   dm_concat=dm_concat,
                   batch_words=batch_words
                   )


def build_doc2vec_model_dbow(size=400, window=8, min_count=1, sample=1e-4, negative=5, workers=None, dm=0, batch_words=10000):
    """
    :param min_count: ignore all words with total frequency lower than this. You have to set this to 1, since the sentence labels only appear once. Setting it any higher than 1 will miss out on the sentences.
    :param window: the maximum distance between the current and predicted word within a sentence. Word2Vec uses a skip-gram model, and this is simply the window size of the skip-gram model.
    :param size: dimensionality of the feature vectors in output. 100 is a good number. If you’re extreme, you can go up to around 400.
    :param sample: threshold for configuring which higher-frequency words are randomly downsampled
    :param negative:
    :param workers: use this many worker threads to train the model
    :return:
    """
    print ">> -----------------------------"
    print "Doc2Vec DBOW model specification:"
    print ">> min_count: " + str(min_count)
    print ">> window: " + str(window)
    print ">> size: " + str(size)
    print ">> negative: " + str(negative)
    print ">> workers: " + str(workers)
    print ">> dm: " + str(dm)
    print ">> batch_words: " + str(batch_words)
    print ">> -----------------------------"

    if workers is None:
        workers = multiprocessing.cpu_count()

    return Doc2Vec(size=size,
                   window=window,
                   min_count=min_count,
                   sample=sample,
                   negative=negative,
                   workers=workers,
                   dm=dm,
                   batch_words=batch_words
                   )


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
        try:
            inferred_vector = model.infer_vector(text.words)
        except:
            inferred_vector = model.infer_vector(text[0])
    else:
        inferred_vector = model.docvecs[text]
    # print ">> inferred vector"
    # print inferred_vector
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.wv.index2word))
    # print ">> sims"
    # print sims
    return sims


def build_matrix_csr(model, sentences, topics):
    topic_list = {}

    vocabulary = model.wv.index2word
    idx_range = len(model.wv.index2word)
    vocab_dict = {vocabulary[i]: i for i in range(idx_range)}

    len_text = len(topics)
    csrmatrix = None
    # print sentences
    for i in tqdm(range(len_text)):
        # print i
        text_nth = [0.0] * idx_range
        # print sentences[i]
        # bow_topic = get_word_distribution(model, [simple_preprocess(sentences[i])])[0]
        # bow_topic = get_word_distribution(model, sentences[i])
        if topics[i] not in topic_list.keys():
            word_dist = get_word_distribution(model, sentences[i])
            topic_list[topics[i]] = {k: v for k, v in word_dist}
            # print ">> dict"
            # print topic_list[topics[i]]

        # print sentences[i]
        for token in simple_preprocess(sentences[i]):
            # print token

            if token in vocab_dict:
                text_nth[vocab_dict[token]] += topic_list[topics[i]][token] if token in topic_list[topics[i]].keys() else 0

                # print ">> changed"
                # print text_nth[vocab_dict[token]]
        # print text_nth
        temp = csr_matrix(text_nth)
        csrmatrix = vstack([csrmatrix, temp]) if csrmatrix is not None else temp
        # print ">> shape"
        # print csrmatrix.shape[0]

    return csrmatrix

def join_tsp(topics, sentences, polarities):
    return topics.to_frame().reset_index(drop=True).join(DataFrame({'TEXT': sentences})).join(polarities.to_frame())

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

# class TestLineSentence(object):
#     def __init__(self, text, label):
#         self.text = text
#
#     def __iter__(self):
#         # yield [TaggedDocument(simple_preprocess(doc), [self.label[idx]]) for idx, doc in enumerate(self.text)]
#         # yield [TaggedDocument(doc, [self.label[idx]]) for idx, doc in enumerate(self.text)]
#         for idx, doc in enumerate(self.text):
#             # print ">> sentences"
#             # print doc
#             # print self.label[idx]
#             yield TaggedDocument(simple_preprocess(doc), [self.label[idx]])