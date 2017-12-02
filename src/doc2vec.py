# coding=utf-8
from gensim.models import word2vec, Doc2Vec
# from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
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


class LabeledLineSentence(object):
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __iter__(self):
        for idx, doc in enumerate(self.text):
            yield TaggedDocument(doc, [self.label[idx]])