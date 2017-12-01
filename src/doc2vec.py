from gensim.models import word2vec
from gensim.models import doc2vec
from tqdm import tqdm

import logging
import random
import os
import sklearn.metrics.pairwise as pairwise


# progress bar
tqdm.pandas(desc="progress-bar")
# format logging configuration
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_sentences(table):
    sentences = []
    for idx, row in table:
        sentences += [row['CLASS'], row['CLEANED_TOKEN']]
    return sentences


def sentences_perm(sentences):
    shuffled = list(sentences)
    random.shuffle(shuffled)
    return shuffled


def get_model(sentences):
    min_count = 1
    size = 200  # or 100
    window = 10
    worker = 5
    range_loop = 1 # 10 or 20

    directory = os.getcwd() + "/../model/doc2vec_" + str(min_count) + "_" + str(size) + "_" + str(window) + ".txt"
    directory = os.path.abspath(directory)

    if os.path.isfile(directory):
        return doc2vec.Doc2Vec.load(directory)

    model = doc2vec.Doc2Vec(min_count=min_count,
                            size=size,
                            window=window,
                            workers=worker
                            )

    model.build_vocab([x[1] for x in tqdm(sentences)])

    for epoch in range(range_loop):
        model.train(tqdm(sentences_perm(sentences)))

    model.save(directory)
    return model


def infer_vector(model, sentence):
    return model.infer_vector(sentence)


def cosine_similarity(sentence1, sentence2):
    return pairwise.cosine_similarity(sentence1, sentence2)


def prediction(model, test):
    for idx, row in test.itterows():
        model.most_similar()
