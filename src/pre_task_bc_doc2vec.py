# https://medium.com/@mishra.thedeepak/doc2vec-in-a-simple-way-fa80bfe81104
from pandas import DataFrame, concat
from doc2vec import LabeledLineSentence
from sklearn import decomposition
from measurements import predict
import matplotlib.pyplot as plt
import doc2vec
import preprocessing
import random
import numpy as np
import svm
import logistic_regression as logres
import multinomial_nb as mnb
import measurements as m


def get_data(filetype='train', dataset='B'):
    return preprocessing.open_preprocess_file(filetype, dataset)


def get_model(raw_data, epoch=200):
    # print raw_data
    random.seed(1)

    # --- get the lables, tweets, and polarities
    topic_lables = raw_data['TOPIC']
    text = raw_data['CLEANED']
    polarity = raw_data['POLARITY']
    # print text

    # --- get total of training instances and topics
    num_train = len(topic_lables)

    print "total data"
    print num_train
    print "total polarity"
    print polarity.value_counts()

    # --- preprocess
    tokens_arr, sents_arr = preprocessing.preprocess(text)
    train_data = preprocessing.join_tsp(topic_lables, sents_arr, polarity)
    sentences = list(LabeledLineSentence(train_data['TEXT'], train_data['TOPIC']))

    model_DM = doc2vec.build_doc2vec_model_dm()
    model_DBOW = doc2vec.build_doc2vec_model_dbow()

    model_DM.build_vocab(sentences)
    model_DBOW.build_vocab(sentences)

    # epoch_loop = 200
    print ">> -----------------------------"
    print "Doc2Vec training"
    print ">> epoch: " + str(epoch)
    print ">> -----------------------------"

    # start training
    for epoch in range(epoch):  # 200
        if epoch % 20 == 0:
            print ('Now training epoch %s' % epoch)

        random.shuffle(sentences)

        model_DM.train(sentences, total_examples=num_train, epochs=epoch)
        model_DM.alpha -= 0.002  # decrease the learning rate
        model_DM.min_alpha = model_DM.alpha  # fix the learning rate, no decay

        model_DBOW.train(sentences, total_examples=num_train, epochs=epoch)
        model_DBOW.alpha -= 0.002  # decrease the learning rate
        model_DBOW.min_alpha = model_DM.alpha  # fix the learning rate, no decay

    # print ">> polarity"
    # print set(polarity)
    # print len(set(polarity))
    # if len(set(polarity[0])) < 3:

    return model_DM, model_DBOW, train_data


def make_doc2vec_test(d2v_model, test_set, mnb=False):
    # --- get the lables, tweets, and polarities
    print "total test polarity"
    print test_set['POLARITY'].value_counts()

    test_tokens, test_sents = preprocessing.preprocess(test_set['CLEANED'])
    test_data = preprocessing.join_tsp(test_set['TOPIC'], test_sents, test_set['POLARITY'])

    if isinstance(test_data['POLARITY'][0], basestring):
        test_data = concat(
            [test_data[test_data.POLARITY == 'positive'], test_data[test_data.POLARITY == 'negative']]).reset_index(
            drop=True)

    csr_matrix_test = doc2vec.build_matrix_csr(model=d2v_model, sentences=test_data['TEXT'], topics=test_data['TOPIC'], mnb=mnb)
    return csr_matrix_test, test_data


def polarity_model(d2v_model, model, train_data, multi, tuning=True):
    # --- get words distribution in for every topic
    if isinstance(train_data['POLARITY'][0], basestring):
        train_data = concat(
            [train_data[train_data.POLARITY == 'positive'], train_data[train_data.POLARITY == 'negative']]).reset_index(
            drop=True)

    train_matrix = doc2vec.build_matrix_csr(model=d2v_model, sentences=train_data['TEXT'], topics=train_data['TOPIC'])

    # train_model = svm.split_and_train(svm_bow, sent_topic['POLARITY'])
    pol_model = None
    if model == 'svm':
        pol_model = svm.split_and_train(train_matrix, train_data['POLARITY'], multi=multi)
    elif model == 'logres':
        pol_model = logres.split_and_train(train_matrix, train_data['POLARITY'], tuning=tuning, multi=multi)
    elif model == 'mnb':
        pol_model = mnb.split_and_train(train_matrix, train_data['POLARITY'], multi=multi)

    return pol_model


def polarity_test(d2v_model, pol_model, dataset):
    csr_matrix_test, test_data = make_doc2vec_test(d2v_model, dataset)
    # --- build svm model >> for polarity
    prediction = predict(csr_matrix_test, test_data['POLARITY'], pol_model)

    return prediction

#
# def svm_polarity_model(model, train_data, multi):
#     svm_train_data = doc2vec.build_matrix_csr(model=model, sentences=train_data['TEXT'],
#                                                  topics=train_data['TOPIC'])
#
#     training_model = svm.split_and_train(svm_train_data, train_data['POLARITY'], multi=multi)
#     return training_model
#
#
# def svm_polarity_test(d2v_model, svm_model, dataset='B'):
#     csr_matrix_test, test_data = make_doc2vec_test(d2v_model, dataset)
#     # --- build svm model >> for polarity
#     prediction = predict(csr_matrix_test, test_data['POLARITY'], svm_model)
#     return prediction
#
#
# def logres_polarity_model(model, train_data, multi):
#     logres_train_data = doc2vec.build_matrix_csr(model=model, sentences=train_data['TEXT'],
#                                                  topics=train_data['TOPIC'])
#
#     training_model = logres.split_and_train(logres_train_data, train_data['POLARITY'], multi=multi)
#     return training_model
#
#
# def logres_polarity_test(d2v_model, logres_model, dataset='B'):
#     csr_matrix_test, test_data = make_doc2vec_test(d2v_model, dataset)
#     # --- build svm model >> for polarity
#     prediction = predict(csr_matrix_test, test_data['POLARITY'], logres_model)
#     return prediction
#
#
# def mnb_polarity_model(model, train_data, multi):
#     mnb_train_data = doc2vec.build_matrix_csr(model=model, sentences=train_data['TEXT'],
#                                                  topics=train_data['TOPIC'], mnb=True)
#
#     # print train_data
#     training_model = mnb.split_and_train(mnb_train_data, train_data['POLARITY'], multi=multi)
#     return training_model
#
#
# def mnb_polarity_test(d2v_model, mnb_model, dataset='B'):
#     csr_matrix_test, test_data = make_doc2vec_test(d2v_model, dataset, mnb=True)
#     # --- build svm model >> for polarity
#     prediction = predict(csr_matrix_test, test_data['POLARITY'], mnb_model)
#     return prediction
