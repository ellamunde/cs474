import random

from pandas import concat

import lda
import preprocessing
import svm
import text_to_vector
from measurements import predict
import logistic_regression as logres
import multinomial_nb as mnb


def get_data(filetype='train', dataset='B'):
    return preprocessing.open_preprocess_file(filetype, dataset)


def get_model(raw_data, passes=20, alpha='auto'):
    random.seed(1)

    # --- get the lables, tweets, and polarities
    all_topics = raw_data['TOPIC']
    text = raw_data['CLEANED']
    polarity = raw_data['POLARITY']

    # --- get total of training instances and topics
    num_train = len(raw_data)
    num_topics = len(all_topics.value_counts())
    print "total data"
    print num_train
    print "total polarity"
    print polarity.value_counts()

    # --- preprocess
    tokens_arr, sents_arr = preprocessing.preprocess(text)

    # --- init vectorizer
    vectorizer = text_to_vector.count_vectorizer()
    # vectorizer = lda.tfidf_vectorizer()
    # vectorizer = lda.fit_to_vectorizer(vectorizer, sents_arr)

    # --- convert text into vectors using vectorizer
    bow_vectorizer = text_to_vector.fit_to_vectorizer(vectorizer, sents_arr)
    # print ">> bow"
    # print bow_vectorizer

    # --- get feature names based on n-grams
    # feature_names = text_to_vector.get_feature_names(vectorizer)

    # --- convert dictionary to id2word
    idvec2word = text_to_vector.map_idvec2word(vectorizer)
    dict_len = len(idvec2word)

    # --- convert bow vectorizer into bow lda
    bow_lda = text_to_vector.convert_to_sparse_bow(bow_vectorizer)
    # print ">> bow lda"
    # print bow_lda

    # --- build lda model >> for topic
    lda_model = lda.build_lda_model(word_bag=bow_lda,
                                    dictionary=idvec2word,
                                    num_topics=num_topics,
                                    alpha=alpha,
                                    passes=passes
                                    )

    train_data = preprocessing.join_tsp(raw_data['TOPIC'], sents_arr, raw_data['POLARITY'])
    topic_ids = lda.assign_topic_to_ldatopic(vectorizer, lda_model, train_data)
    topn = dict_len
    topic_words_dist = lda.get_words_topic(lda_model,
                                           topic_ids,
                                           topn
                                           )

    return lda_model, vectorizer, train_data, all_topics, topic_words_dist, topic_ids


def make_lda_train(lda_model, vectorizer, topic_words_dist, train_data, topic_ids):
    train_matrix = lda.build_matrix_csr(vectorizer=vectorizer,
                                            lda_model=lda_model,
                                            topic_words_dist=topic_words_dist,
                                            map_topic_id=topic_ids,
                                            dataset=train_data
                                            )
    return train_matrix


def make_lda_test(lda_model, vectorizer, topic_words_dist, test_set, topic_ids):
    # --- get the lables, tweets, and polarities
    print "total test polarity"
    print test_set['POLARITY'].value_counts()

    if isinstance(test_set['POLARITY'][0], basestring):
        test_set = concat([test_set[test_set.POLARITY == 'positive'],
                           test_set[test_set.POLARITY == 'negative']]).reset_index(drop=True)

    # print test_data
    test_tokens, test_sents = preprocessing.preprocess(test_set['CLEANED'])
    test_set = preprocessing.join_tsp(test_set['TOPIC'], test_sents, test_set['POLARITY'])

    csr_matrix_test = lda.build_matrix_csr(vectorizer=vectorizer,
                                           lda_model=lda_model,
                                           topic_words_dist=topic_words_dist,
                                           map_topic_id=topic_ids,
                                           dataset=test_set
                                           )
    return csr_matrix_test, test_set


def polarity_model(lda_model, model, vectorizer, topic_words_dist, train_data, multi, map_topic_id, tuning=True):
    # --- get words distribution in for every topic
    if isinstance(train_data['POLARITY'][0], basestring):
        train_data = concat(
            [train_data[train_data.POLARITY == 'positive'], train_data[train_data.POLARITY == 'negative']]).reset_index(
            drop=True)

    train_matrix = make_lda_train(vectorizer=vectorizer,
                                  lda_model=lda_model,
                                  topic_words_dist=topic_words_dist,
                                  train_data=train_data,
                                  topic_ids=map_topic_id
                                  )
    print len(train_data)
    # train_model = svm.split_and_train(svm_bow, sent_topic['POLARITY'])
    pol_model = None
    if model == 'svm':
        pol_model = svm.split_and_train(train_matrix, train_data['POLARITY'], multi=multi)
    elif model == 'logres':
        pol_model = logres.split_and_train(train_matrix, train_data['POLARITY'], tuning=tuning, multi=multi)
    elif model == 'mnb':
        pol_model = mnb.split_and_train(train_matrix, train_data['POLARITY'], multi=multi)

    return pol_model


def polarity_test(lda_model, svm_model, vectorizer, topic_words_dist, dataset, map_topic_id):
    csr_matrix_test, sent_topic_test = make_lda_test(lda_model, vectorizer, topic_words_dist, dataset, topic_ids=map_topic_id)
    prediction = predict(csr_matrix_test, sent_topic_test['POLARITY'], svm_model)
    return prediction
#
#
# def svm_polarity_model(lda_model, vectorizer, topic_words_dist, traindata, multi):
#     # --- get words distribution in for every topic
#     svm_train_matrix = make_lda_train(vectorizer=vectorizer,
#                                       lda_model=lda_model,
#                                       topic_words_dist=topic_words_dist,
#                                       train_data=traindata
#                                       )
#
#     # train_model = svm.split_and_train(svm_bow, sent_topic['POLARITY'])
#     svm_pol_model = svm.split_and_train(svm_train_matrix, traindata['POLARITY'], multi=multi)
#     return svm_pol_model
#
#
# def svm_polarity_test(lda_model, svm_model, vectorizer, topic_words_dist, dataset='B'):
#     csr_matrix_test, sent_topic_test = make_lda_test(lda_model, vectorizer, topic_words_dist, dataset=dataset)
#     # --- build svm model >> for polarity
#     # prediction_res = predict(csr_matrix_test, test_data['POLARITY'], train_model)
#     prediction = predict(csr_matrix_test, sent_topic_test['POLARITY'], svm_model)
#     return prediction
#
#
# def logres_polarity_model(lda_model, vectorizer, topic_words_dist, traindata, multi, tuning=True, ):
#     # --- get words distribution in for every topic
#     logres_train_matrix = make_lda_train(vectorizer=vectorizer,
#                                          lda_model=lda_model,
#                                          topic_words_dist=topic_words_dist,
#                                          train_data=traindata
#                                          )
#
#     pol_model = logres.split_and_train(logres_train_matrix, traindata['POLARITY'], tuning=tuning, multi=multi)
#     return pol_model
#
#
# def logres_polarity_test(lda_model, logres_model, vectorizer, topic_words_dist, dataset='B'):
#     csr_matrix_test, sent_topic_test = make_lda_test(lda_model, vectorizer, topic_words_dist, dataset=dataset)
#     prediction = predict(csr_matrix_test, sent_topic_test['POLARITY'], logres_model)
#     return prediction
#
#
# def multinomial_nb_polarity_model(lda_model, vectorizer, topic_words_dist, traindata, multi):
#     # --- get words distribution in for every topic
#     svm_train_matrix = make_lda_train(vectorizer=vectorizer,
#                                       lda_model=lda_model,
#                                       topic_words_dist=topic_words_dist,
#                                       train_data=traindata
#                                       )
#
#     # train_model = svm.split_and_train(svm_bow, sent_topic['POLARITY'])
#     mnb_pol_model = mnb.split_and_train(svm_train_matrix, traindata['POLARITY'], multi=multi)
#     return mnb_pol_model
#
#
# def multinomial_nb_polarity_test(lda_test, mnb_model, vectorizer, topic_words_dist, dataset='B'):
#     csr_matrix_test, sent_topic_test = make_lda_test(lda_test, vectorizer, topic_words_dist, dataset=dataset)
#     prediction = predict(csr_matrix_test, sent_topic_test['POLARITY'], mnb_model)
#     return prediction
